"""The ingestion pipeline: scan -> plan -> parse -> segment -> enrich -> index.

This is where the frame's promises are kept rather than described. Four of them
live here and nowhere else:

**The plan is computed before any work.** ``plan()`` diffs the corpus against
the ledger and returns a list of ``PlannedChange``. A build can therefore report
"10 of 510 documents" before spending anything, and a surprising number can be
questioned before rather than after.

**Per-stage invalidation.** Each document's ledger record holds a fingerprint
key per stage. Changing the reranker reprocesses nothing; changing the segmenter
reprocesses segment onward; changing one enricher reruns that enricher only.

**Deletions are handled.** Documents in the ledger but not in the corpus have
their units deleted from every index. A cache alone cannot do this, and without
it a removed document answers queries forever.

**Contract enforcement.** Parser and segmenter output is checked against
``eval.checks`` before it is allowed downstream, because a broken span here
becomes a wrong citation three stages later with nothing to attribute it to.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indexer.core.accounting import Accountant, CacheOutcome, InMemoryAccountant
from indexer.core.cache import CacheKey, CacheStore, cache_key
from indexer.core.document import ParsedDocument, SourceDocument, content_metadata
from indexer.core.errors import ContractViolation, DocumentError, EnrichmentIncomplete
from indexer.core.ids import ContentHash, DocumentId, hash_obj, hash_text
from indexer.core.ledger import ChangeKind, DocumentRecord, Ledger, PlannedChange, diff_units
from indexer.core.manifest import BuildManifest, CorpusStats, IndexStats
from indexer.core.stages import (
    CorpusScanner,
    EnrichContext,
    Enricher,
    Flushable,
    Index,
    Parser,
    Segmenter,
    StageContext,
    enrich_input_hash,
    parse_cache_scope,
)
from indexer.core.unit import EnrichedUnit, Enrichment, Unit
from indexer.eval.checks import check_parsed_document, check_units
from indexer.io import atomic_write
from indexer.pipeline.codec import (
    decode_enrichment,
    decode_metadata,
    decode_parsed_document,
    decode_units,
    encode_enrichment,
    encode_metadata,
    encode_parsed_document,
    encode_units,
)
from indexer.pipeline.stores import CacheRefs, UnitStore, sweep_cache
from indexer.textutil import detect_language

__all__ = [
    "BuildResult",
    "IngestionPipeline",
    "PrefillReport",
    "cached_parse",
    "parse_cache_key",
    "rebind_parsed",
    "strip_identity",
    "with_document_facts",
]


@dataclass(slots=True)
class _Pending:
    """Work done since the last checkpoint, not yet durable in the ledger."""

    records: list[DocumentRecord] = field(default_factory=list)
    removals: list[DocumentId] = field(default_factory=list)
    refs: list[tuple[DocumentId, set[str]]] = field(default_factory=list)
    #: Cache keys some document stopped using. Deleted at the end of the build
    #: if nothing else uses them.
    purge_candidates: set[str] = field(default_factory=set)


@dataclass(slots=True)
class PrefillReport:
    """What ``IngestionPipeline.prefill`` did."""

    rounds: int = 0
    requests: int = 0
    #: Enrichments written to the cache, ready for the build.
    cached: int = 0
    #: Units whose answer failed; the build will call for them again.
    failed: int = 0
    cost_usd: float = 0.0
    documents_failed: int = 0
    by_enricher: dict[str, int] = field(default_factory=dict)


#: A stage key recorded for an enricher that left some units of a document
#: without its enrichment. It matches no fingerprint, so the next build's plan
#: restages the document: cached units hit, the failed ones are asked again.
INCOMPLETE = "incomplete"


@dataclass(slots=True)
class BuildResult:
    manifest: BuildManifest
    plan: list[PlannedChange] = field(default_factory=list)
    failures: list[DocumentError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        c = self.manifest.corpus
        return (
            f"{c.documents_total} docs "
            f"(+{c.documents_added} ~{c.documents_changed} "
            f"restaged {c.documents_restaged} skip {c.documents_unchanged} "
            f"-{c.documents_removed} fail {c.documents_failed}), "
            f"{c.units_written} units written, {c.units_deleted} deleted, "
            f"{self.manifest.elapsed_wall_ms / 1000:.1f}s "
            f"({self.manifest.accounted_fraction:.0%} in stages), "
            f"${self.manifest.total_cost_usd:.4f}"
        )


class IngestionPipeline:
    """Wires the four ingestion stages over a corpus.

    Stages are passed in already constructed. Building them from config is the
    assembler's job (``indexer.pipeline.build``); keeping construction out of
    here means the pipeline can be driven with hand-made stages in a test.
    """

    def __init__(
        self,
        *,
        scanner: CorpusScanner,
        parser: Parser,
        segmenter: Segmenter,
        enrichers: Sequence[Enricher],
        indexes: Mapping[str, Index],
        ledger: Ledger,
        unit_store: UnitStore,
        cache: CacheStore,
        config: Mapping[str, Any] | None = None,
        config_hash: ContentHash | None = None,
        library_version: str = "0.1.0",
        strict_contracts: bool = True,
        enrich_enabled: bool = True,
        segment_enabled: bool = True,
        parse_enabled: bool = True,
        on_document_error: str = "skip",
        index_batch_size: int = 128,
        enrich_batch_size: int = 16,
        enrich_max_concurrency: int = 1,
        enrich_on_error: str = "fail",
        checkpoint_every: int = 200,
        cache_refs: CacheRefs | None = None,
        purge_cache: bool = True,
        manifest_dir: str | Path | None = None,
    ) -> None:
        self.scanner = scanner
        #: Where each build's manifest is written, as ``<build_id>.json`` and
        #: ``latest.json``. The manifest said it was written there; nothing did.
        self.manifest_dir = Path(manifest_dir) if manifest_dir is not None else None
        #: Which cache entries each document uses, so that removing or editing
        #: a document can remove what it left in the cache. None disables it.
        self.cache_refs = cache_refs
        self.purge_cache = purge_cache
        self.parser = parser
        self.segmenter = segmenter
        self.enrichers = list(enrichers)
        self.indexes = dict(indexes)
        self.ledger = ledger
        self.unit_store = unit_store
        self.cache = cache
        self.config = dict(config or {})
        self.config_hash = config_hash or hash_obj(self.config)
        self.library_version = library_version
        self.strict_contracts = strict_contracts
        self.enrich_enabled = enrich_enabled
        self.segment_enabled = segment_enabled
        self.parse_enabled = parse_enabled
        self.on_document_error = on_document_error
        self.index_batch_size = index_batch_size
        #: Units per enricher call, calls in flight per batch, and what a
        #: failed enrichment does: ``fail`` fails the document, ``skip`` indexes
        #: its units without that enrichment and retries it next build. All
        #: three were configurable and none was read.
        self.enrich_batch_size = max(1, enrich_batch_size)
        self.enrich_max_concurrency = max(1, enrich_max_concurrency)
        if enrich_on_error not in ("fail", "skip"):
            raise ValueError(f"enrich_on_error must be fail or skip, not {enrich_on_error!r}")
        self.enrich_on_error = enrich_on_error
        # Documents processed between durability checkpoints. This is the unit
        # of resumability: a build that dies resumes at the last checkpoint, not
        # at the last document, because a document is only recorded as done once
        # the indexes holding it are durable.
        self.checkpoint_every = max(1, checkpoint_every)

    # ------------------------------------------------------------------ plan

    def stage_keys(self) -> dict[str, str]:
        """Fingerprint key per stage, as stored in the ledger.

        Per stage, and per *enricher* and per *index* within their stages: that
        granularity is the whole point. One blob would make every change a full
        rebuild.
        """
        keys = {
            "parse": self.parser.fingerprint().key() if self.parse_enabled else "disabled",
            "segment": self.segmenter.fingerprint().key() if self.segment_enabled else "disabled",
        }
        for e in self.enrichers if self.enrich_enabled else []:
            keys[f"enrich:{e.name}"] = e.fingerprint().key()
        if not (self.enrich_enabled and self.enrichers):
            keys["enrich"] = "disabled"
        for name, idx in self.indexes.items():
            keys[f"index:{name}"] = idx.fingerprint().key()
        return keys

    def plan(
        self, *, scanned: Mapping[DocumentId, SourceDocument] | None = None
    ) -> list[PlannedChange]:
        """Diff the corpus against the ledger. No work is done here."""
        current = (
            dict(scanned)
            if scanned is not None
            else {d.document_id: d for d in self.scanner.scan()}
        )
        keys = self.stage_keys()
        plan: list[PlannedChange] = []

        for doc_id, doc in current.items():
            prior = self.ledger.get(doc_id)
            if prior is None:
                plan.append(PlannedChange(doc_id, ChangeKind.ADDED, tuple(keys), "not seen before"))
                continue
            if prior.content_hash != doc.content_hash:
                plan.append(
                    PlannedChange(doc_id, ChangeKind.CHANGED, tuple(keys), "content changed", prior)
                )
                continue
            if prior.metadata_hash != metadata_hash(doc):
                plan.append(
                    PlannedChange(
                        doc_id, ChangeKind.CHANGED, tuple(keys), "metadata changed", prior
                    )
                )
                continue
            stale = [k for k, v in keys.items() if prior.stage_keys.get(k) != v]
            if stale:
                retry = all(prior.stage_keys.get(k) == INCOMPLETE for k in stale)
                what = "retrying incomplete enrichment" if retry else "stage fingerprint changed"
                plan.append(
                    PlannedChange(
                        doc_id,
                        ChangeKind.RESTAGED,
                        tuple(stale),
                        f"{what}: {', '.join(sorted(stale))}",
                        prior,
                    )
                )
                continue
            plan.append(PlannedChange(doc_id, ChangeKind.UNCHANGED, (), "up to date", prior))

        for gone in self.ledger.document_ids() - set(current):
            plan.append(
                PlannedChange(
                    gone, ChangeKind.REMOVED, (), "no longer in corpus", self.ledger.get(gone)
                )
            )
        return plan

    # ----------------------------------------------------------------- build

    def build(
        self,
        *,
        plan: Sequence[PlannedChange] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> BuildResult:
        started = time.perf_counter()
        accountant = InMemoryAccountant()
        ctx = StageContext(cache=self.cache, accountant=accountant)
        manifest = BuildManifest.start(
            config=self.config,
            config_hash=self.config_hash,
            library_version=self.library_version,
            project=str(self.config.get("project", {}).get("name", "")),
        )
        manifest.stage_fingerprints = {k: {"key": v} for k, v in self.stage_keys().items()}
        manifest.disabled_stages = self._disabled()
        self.ledger.begin_build(manifest.build_id)

        # Scan once. `plan()` already read and hashed every document; scanning
        # again to build this map doubles the corpus read for no new information.
        scanned = {d.document_id: d for d in self.scanner.scan()}
        work = list(plan if plan is not None else self.plan(scanned=scanned))
        by_id = scanned
        stats = CorpusStats(documents_total=len(by_id))
        failures: list[DocumentError] = []
        confidences: list[float] = []
        pending = _Pending()

        for change in work:
            counter = {
                ChangeKind.ADDED: "documents_added",
                ChangeKind.CHANGED: "documents_changed",
                ChangeKind.RESTAGED: "documents_restaged",
                ChangeKind.UNCHANGED: "documents_unchanged",
                ChangeKind.REMOVED: "documents_removed",
            }[change.kind]
            setattr(stats, counter, getattr(stats, counter) + 1)

            if change.kind is ChangeKind.UNCHANGED:
                stats.units_total += len(change.prior.unit_ids) if change.prior else 0
                continue

            if change.kind is ChangeKind.REMOVED:
                if change.prior:
                    stats.units_deleted += self._delete_units(list(change.prior.unit_ids), ctx)
                # Deferred to the checkpoint, like every other ledger write. The
                # units were deleted from indexes in memory; forgetting the
                # document before those deletions are durable meant a crash left
                # its units on disk with no ledger entry to ever remove them.
                pending.removals.append(change.document_id)
                continue

            doc = by_id[change.document_id]
            if progress:
                progress(f"{change.kind.value:>9} {doc.source_uri}")
            try:
                parsed, _units, enriched, keys_used, incomplete = self._process(doc, ctx)
            except DocumentError as exc:
                failures.append(exc)
                stats.documents_failed += 1
                setattr(stats, counter, getattr(stats, counter) - 1)
                if self.on_document_error == "fail":
                    raise
                continue

            confidences.append(parsed.reading_order_confidence)
            stats.bytes_parsed += len(parsed.text)
            stats.enrichments_failed += sum(incomplete.values())
            new_ids = [u.unit_id for u in enriched]
            prior_ids = list(change.prior.unit_ids) if change.prior else []
            _, to_delete = diff_units(prior_ids, new_ids)

            # Restaging an index (a changed embedding model) must rewrite every
            # unit, not just the changed ones -- the units are the same but their
            # representation in that index is not. Otherwise a unit is written
            # when its *record* changed, not only when its id is new: ids come
            # from text, so a paragraph inserted above a unit keeps its id and
            # moves its span, and rewriting new ids only left every such unit
            # citing the wrong offset.
            index_restaged = change.kind is ChangeKind.RESTAGED and any(
                s.startswith("index:") for s in change.stages
            )
            full_rewrite = change.kind is ChangeKind.ADDED or not prior_ids or index_restaged
            to_write = (
                enriched
                if full_rewrite
                else [u for u in enriched if not self.unit_store.is_current(u)]
            )

            if to_delete:
                stats.units_deleted += self._delete_units(list(to_delete), ctx)
            written = 0
            if to_write:
                self.unit_store.put_many(to_write)
                for i in range(0, len(to_write), self.index_batch_size):
                    batch = to_write[i : i + self.index_batch_size]
                    batch_written = 0
                    for name, idx in self.indexes.items():
                        with accountant.measure(idx.fingerprint()) as run:
                            receipt = idx.upsert(batch, ctx)
                            run.items_in = len(batch)
                            run.items_out = receipt.written
                            run.cost_usd = receipt.cost_usd
                            run.attrs = {"index": name}
                        batch_written = max(batch_written, receipt.written)
                    written += batch_written
            # Counted as the indexes saw it: a unit whose neighbour changed is
            # stored again (its links moved) but no index rewrites it, and no
            # vector is recomputed -- that is the reuse this number reports.
            stats.units_written += written
            stats.units_total += len(enriched)
            stats.units_reused_from_cache += len(enriched) - written

            # Buffered, not committed. A ledger record claims a document is
            # done; committing it before the indexes holding that document are
            # durable is a lie the next build believes -- it skips the document
            # and the units are nowhere. That is not hypothetical: killing a
            # build mid-run left a ledger asserting 493 processed documents over
            # indexes that held none, and the resumed build skipped all 493.
            pending.records.append(
                DocumentRecord(
                    document_id=doc.document_id,
                    source_uri=doc.source_uri,
                    content_hash=doc.content_hash,
                    unit_ids=tuple(new_ids),
                    stage_keys=self._record_keys(incomplete),
                    build_id=manifest.build_id,
                    updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    metadata_hash=metadata_hash(doc),
                )
            )
            pending.refs.append((doc.document_id, keys_used))
            if len(pending.records) >= self.checkpoint_every:
                self._checkpoint(pending, accountant)

        # Final checkpoint: everything still buffered becomes durable, and only
        # then are the remaining documents recorded as done.
        self._checkpoint(pending, accountant)
        stats.cache_entries_purged = self._purge(pending.purge_candidates)

        if confidences:
            stats.mean_reading_order_confidence = sum(confidences) / len(confidences)
        manifest.corpus = stats
        manifest.indexes = [
            IndexStats(
                name=n,
                kind=str(i.kind),
                impl=i.fingerprint().impl,
                unit_count=i.stats().unit_count,
                size_bytes=i.stats().size_bytes,
                detail=i.stats().detail,
            )
            for n, i in self.indexes.items()
        ]
        manifest.stage_totals = accountant.by_stage()
        manifest.absorb(accountant.runs())
        manifest.finish(elapsed_wall_ms=(time.perf_counter() - started) * 1000.0)
        self.ledger.commit_build(manifest.build_id)
        if self.manifest_dir is not None:
            data = manifest.to_json().encode("utf-8")
            atomic_write(self.manifest_dir / f"{manifest.build_id}.json", data)
            atomic_write(self.manifest_dir / "latest.json", data)
        return BuildResult(manifest=manifest, plan=work, failures=failures)

    # ------------------------------------------------------------- internals

    def _checkpoint(self, pending: _Pending, accountant: Accountant | None = None) -> None:
        """Make index state durable, then record the documents it covers.

        The order is the whole point. Flushing after committing the ledger makes
        the ledger a claim about work that may not survive the process; flushing
        first makes every committed record backed by bytes on disk. A crash in
        between costs a redundant reprocess of the checkpoint's documents, which
        is the correct failure: too much work, never too little.

        The flush is measured. For most indexes it is a file write, but an index
        that defers real work to commit time -- an SVD embedder fits its
        projection there -- can spend more in flush than in every stage
        combined. Leaving that outside the accounting reported such a build as
        "6% in stages" with no indication of where the rest went.
        """
        for name, idx in self.indexes.items():
            if not isinstance(idx, Flushable):
                continue
            if accountant is None:
                idx.flush()
                continue
            fp = replace(idx.fingerprint(), stage="index.flush")
            with accountant.measure(fp) as run:
                idx.flush()
                run.attrs = {"index": name}
                run.items_in = len(pending.records)
        self.unit_store.flush()
        for record in pending.records:
            self.ledger.put(record)
        for document_id in pending.removals:
            self.ledger.delete(document_id)
        if self.cache_refs is not None:
            for document_id, keys in pending.refs:
                pending.purge_candidates |= self.cache_refs.record(document_id, keys)
            for document_id in pending.removals:
                pending.purge_candidates |= self.cache_refs.remove(document_id)
        pending.records.clear()
        pending.removals.clear()
        pending.refs.clear()

    def _purge(self, candidates: set[str]) -> int:
        """Delete cache entries that no current document uses any more.

        The candidates are the entries removed documents used, and the entries
        the previous version of an edited document used. Each is deleted only if
        nothing else references it: the cache is shared by design, and a parse
        another copy of the file still uses must survive. Runs after the final
        checkpoint, so references are durable before anything is deleted.
        """
        if self.cache_refs is None or not self.purge_cache or not candidates:
            return 0
        doomed = self.cache_refs.unreferenced(candidates)
        for key in doomed:
            self.cache.delete(key)
        return len(doomed)

    def sweep_cache(self) -> int:
        """Delete every cache entry no current document references.

        The full collection, for entries no build recorded: written by a build
        that died before its checkpoint, or by a version that kept no
        references. Returns the number of entries deleted.
        """
        if self.cache_refs is None:
            raise RuntimeError("sweep_cache needs cache references; none are configured")
        return sweep_cache(self.cache, self.cache_refs.all_keys())

    def _disabled(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if not self.parse_enabled:
            out["parse"] = "config: parse.enabled=false (passthrough: bytes as one block)"
        if not self.segment_enabled:
            out["segment"] = "config: segment.enabled=false (one unit per document)"
        if not self.enrich_enabled or not self.enrichers:
            out["enrich"] = "config: enrich disabled (units indexed without context)"
        return out

    def _delete_units(self, unit_ids: list[Any], ctx: StageContext) -> int:
        for idx in self.indexes.values():
            idx.delete(unit_ids, ctx)
        self.unit_store.delete_many(unit_ids)
        return len(unit_ids)

    def _process(
        self, doc: SourceDocument, ctx: StageContext
    ) -> tuple[ParsedDocument, list[Unit], list[EnrichedUnit], set[str], dict[str, int]]:
        """Run the stages for one document. Also returns the cache keys it used,
        so that removing the document later can remove what it left behind, and
        the units each enricher failed on, so the next build can retry them."""
        keys: set[str] = set()
        incomplete: dict[str, int] = {}
        parsed = with_document_facts(self._parse(doc, ctx, keys))
        units = self._segment(parsed, ctx, keys)
        enriched = self._enrich(parsed, units, ctx, keys, incomplete)
        return parsed, units, enriched, keys, incomplete

    def _record_keys(self, incomplete: Mapping[str, int]) -> dict[str, str]:
        """The stage keys a document's ledger record carries.

        An enricher that failed on some of its units is recorded as
        ``INCOMPLETE`` rather than with its fingerprint, so the next plan sees a
        stale stage and restages the document -- the enrichments that were made
        come from the cache, and only the failed ones are asked again. Without
        it a transient API error became a permanent gap in the index.
        """
        keys = self.stage_keys()
        for name, failed in incomplete.items():
            if failed:
                keys[f"enrich:{name}"] = INCOMPLETE
        return keys

    def _parse(
        self, doc: SourceDocument, ctx: StageContext, keys: set[str] | None = None
    ) -> ParsedDocument:
        """Parse, with the output cached by *content* and identity stamped after.

        Content-addressed because an archive is full of byte-identical files --
        the same contract attached to fifty emails -- and a layout or OCR parse
        is the slowest stage there is. But the cached value carries no
        identity: the first document's id, URI and metadata used to come back
        for the second, so both collapsed into one document and the second
        tenant's copy was unreachable.
        """
        fp = self.parser.fingerprint()
        key = parse_cache_key(self.parser, doc)
        if keys is not None:
            keys.add(key)
        cached = ctx.cache.get(key)
        if cached is not None:
            with ctx.accountant.measure(fp) as run:
                run.cache = CacheOutcome.HIT
                run.input_hash = doc.content_hash
                payload = json.loads(cached)
                return rebind_parsed(
                    decode_parsed_document(payload),
                    doc,
                    decode_metadata(payload.get("parser_metadata", {})),
                )

        with ctx.accountant.measure(fp) as run:
            run.cache = CacheOutcome.MISS
            run.input_hash = doc.content_hash
            try:
                parsed = self.parser.parse(doc, ctx)
            except Exception as exc:
                raise DocumentError(doc.document_id, "parse", str(exc)) from exc
            problems = check_parsed_document(parsed)
            if problems:
                msg = f"{len(problems)} contract violation(s): " + "; ".join(problems[:3])
                if self.strict_contracts:
                    raise ContractViolation(f"{self.parser.fingerprint().key()}: {msg}")
                run.attrs = {"contract_warnings": problems[:3]}
            run.output_hash = parsed.content_hash
            run.items_out = len(parsed.blocks)
            # What the parser added to the scanner's metadata, from the bytes.
            # Stored apart from identity so a hit can rebuild exactly this.
            added = {
                k: v
                for k, v in parsed.metadata.items()
                if k not in doc.metadata or doc.metadata[k] != v
            }
            anonymous = strip_identity(parsed)
            payload = encode_parsed_document(anonymous)
            payload["parser_metadata"] = encode_metadata(added)
            ctx.cache.put(key, json.dumps(payload).encode("utf-8"))
        # Identity is stamped on the miss path too, so every parser -- including
        # one that forgets to copy scanner metadata -- produces the same result
        # whether or not the cache answered.
        return rebind_parsed(parsed, doc, added)

    def _segment(
        self, parsed: ParsedDocument, ctx: StageContext, keys: set[str] | None = None
    ) -> list[Unit]:
        fp = self.segmenter.fingerprint()
        # The segmenter's input is the whole parsed document -- blocks, kinds,
        # pages, identity and metadata, all of which reach the units -- not just
        # its text. Keying on text alone served one document's units (ids,
        # tenant and all) to every other document with the same text.
        input_hash = hash_obj(encode_parsed_document(parsed))
        key = cache_key(fp, input_hash)
        if keys is not None:
            keys.add(key)
        cached = ctx.cache.get(key)
        if cached is not None:
            with ctx.accountant.measure(fp) as run:
                run.cache = CacheOutcome.HIT
                run.input_hash = input_hash
                return decode_units(json.loads(cached))

        with ctx.accountant.measure(fp) as run:
            run.cache = CacheOutcome.MISS
            run.input_hash = input_hash
            try:
                units = list(self.segmenter.segment(parsed, ctx))
            except Exception as exc:
                raise DocumentError(parsed.document_id, "segment", str(exc)) from exc
            problems = check_units(units, parsed)
            if problems:
                msg = f"{len(problems)} contract violation(s): " + "; ".join(problems[:3])
                if self.strict_contracts:
                    raise ContractViolation(f"{fp.key()}: {msg}")
                run.attrs = {"contract_warnings": problems[:3]}
            run.items_out = len(units)
            ctx.cache.put(key, json.dumps(encode_units(units)).encode("utf-8"))
        return units

    def _enrich(
        self,
        parsed: ParsedDocument,
        units: list[Unit],
        ctx: StageContext,
        used: set[str] | None = None,
        incomplete: dict[str, int] | None = None,
    ) -> list[EnrichedUnit]:
        enriched = [EnrichedUnit(unit=u) for u in units]
        if not self.enrich_enabled:
            return enriched

        prior: dict[Any, dict[str, Enrichment]] = {}
        for enricher in self.enrichers:
            keys = self._enrich_keys(enricher, parsed, enriched, prior)
            if used is not None:
                used.update(keys)
            results, failed = self._run_enricher(
                enricher, parsed, units, enriched, keys, prior, ctx
            )
            if failed and incomplete is not None:
                incomplete[enricher.name] = incomplete.get(enricher.name, 0) + failed
            for i, e in results.items():
                enriched[i] = enriched[i].with_enrichment(e)
                prior.setdefault(enriched[i].unit_id, {})[e.enricher] = e
        return enriched

    @staticmethod
    def _enrich_keys(
        enricher: Enricher,
        parsed: ParsedDocument,
        enriched: Sequence[EnrichedUnit],
        prior: Mapping[Any, Mapping[str, Enrichment]],
    ) -> list[str]:
        """One cache key per unit. The key covers what the enricher reads: by
        default the whole unit (metadata and section path included), the parent
        document for a wider scope, and earlier enrichers' output. A unit-scoped
        enricher still survives edits elsewhere in its document, which is what
        keeps an edit's blast radius proportional to the edit."""
        fp = enricher.fingerprint()
        return [
            cache_key(fp, enrich_input_hash(enricher, eu.unit, parsed, prior.get(eu.unit_id, {})))
            for eu in enriched
        ]

    def _run_enricher(
        self,
        enricher: Enricher,
        parsed: ParsedDocument,
        units: list[Unit],
        enriched: Sequence[EnrichedUnit],
        keys: Sequence[str],
        prior: Mapping[Any, Mapping[str, Enrichment]],
        ctx: StageContext,
    ) -> tuple[dict[int, Enrichment], int]:
        """One enricher over one document: cached results, then the rest in
        batches. Returns the enrichments by unit position, and how many units
        were left without one (``enrich.on_error: skip``)."""
        fp = enricher.fingerprint()
        results: dict[int, Enrichment] = {}
        todo: list[int] = []
        for i, key in enumerate(keys):
            cached = ctx.cache.get(key)
            if cached is None:
                todo.append(i)
            else:
                results[i] = decode_enrichment(json.loads(cached))
        if not todo:
            with ctx.accountant.measure(fp) as run:
                run.cache = CacheOutcome.HIT
                run.items_in = run.items_out = len(enriched)
            return results, 0

        # One computation per distinct key. The key is the enricher's whole
        # input, so units that share one share the output by contract: every
        # unit of a document under a document-level classifier, the footer
        # repeated on every page. A classifier made one call per *unit* before.
        first: dict[str, int] = {}
        for i in todo:
            first.setdefault(keys[i], i)
        reps = list(first.values())
        context = EnrichContext(
            document=parsed,
            units=units,
            stage=ctx,
            prior=dict(prior),
            max_concurrency=self.enrich_max_concurrency,
        )
        failed: set[int] = set()
        for start in range(0, len(reps), self.enrich_batch_size):
            chunk = reps[start : start + self.enrich_batch_size]
            with ctx.accountant.measure(fp) as run:
                run.cache = CacheOutcome.MISS
                run.items_in = len(chunk)
                errors: list[str] = []
                produced: Sequence[Enrichment | None]
                try:
                    produced = enricher.enrich([enriched[i].unit for i in chunk], context)
                except EnrichmentIncomplete as exc:
                    produced, errors = exc.results, exc.errors
                    run.tokens_in += exc.tokens_in
                    run.tokens_out += exc.tokens_out
                    run.cost_usd += exc.cost_usd
                except Exception as exc:
                    # A failure that will recur on every call -- a bad key, a
                    # model that does not exist -- stops the build instead of
                    # being recorded once per unit of the whole corpus.
                    if getattr(exc, "fatal", False):
                        raise
                    if self.enrich_on_error == "fail":
                        raise DocumentError(
                            parsed.document_id, f"enrich:{fp.impl}", str(exc)
                        ) from exc
                    produced, errors = [None] * len(chunk), [f"{type(exc).__name__}: {exc}"]
                if len(produced) != len(chunk):
                    raise ContractViolation(
                        f"{fp.key()}: returned {len(produced)} enrichments for "
                        f"{len(chunk)} units; the contract is one per input, in order"
                    )
                if errors:
                    run.error = "; ".join(errors[:3])[:500]
                    if self.enrich_on_error == "fail":
                        raise DocumentError(parsed.document_id, f"enrich:{fp.impl}", run.error)
                done = 0
                for i, e in zip(chunk, produced, strict=True):
                    if e is None:
                        failed.add(i)
                        continue
                    done += 1
                    results[i] = e
                    run.cost_usd += e.cost_usd
                    run.tokens_in += e.tokens_in
                    run.tokens_out += e.tokens_out
                    ctx.cache.put(keys[i], json.dumps(encode_enrichment(e)).encode("utf-8"))
                run.items_out = done
        for i in todo:
            rep = first[keys[i]]
            if rep in results:
                results[i] = results[rep]
        missing = sum(1 for i in todo if i not in results)
        return results, missing

    # ------------------------------------------------------------- prefill

    def prefill(
        self,
        *,
        enrichers: Sequence[str] | None = None,
        poll_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        progress: Callable[[str], None] | None = None,
        max_rounds: int = 8,
    ) -> PrefillReport:
        """Make the model calls the next build needs through the Message Batches
        API -- at half the price -- and leave the answers in the cache.

        For a first build of an archive, or a re-contextualisation after a
        prompt change, where nobody is waiting: the build that follows finds
        every model-backed enrichment cached and makes no calls. The requests are
        the ones ``enrich`` would send (``ModelEnricher.requests_for``), so a
        batch-built index is the index a live build would have made.

        An enricher that reads an earlier one's output -- a field extractor
        reading the classifier's document type -- needs that output first, so
        the work goes in rounds: each round submits, for every document, the
        first model call still missing in its chain.
        """
        report = PrefillReport()
        ctx = StageContext(cache=self.cache, accountant=InMemoryAccountant())
        if not self.enrich_enabled:
            return report
        scanned = {d.document_id: d for d in self.scanner.scan()}
        work = [
            c
            for c in self.plan(scanned=scanned)
            if c.kind in (ChangeKind.ADDED, ChangeKind.CHANGED, ChangeKind.RESTAGED)
        ]
        done: set[str] = set()
        asked: set[tuple[str, str]] = set()
        for _ in range(max_rounds):
            pending: list[_BatchWork] = []
            for change in work:
                if change.document_id in done:
                    continue
                try:
                    item = self._next_model_call(scanned[change.document_id], ctx, enrichers)
                except DocumentError:
                    done.add(change.document_id)
                    report.documents_failed += 1
                    continue
                if item is None:
                    done.add(change.document_id)
                    continue
                # Asked once already and still missing: those answers failed.
                # The build will ask again; another round would fail the same way.
                if (change.document_id, item.enricher.name) in asked:
                    done.add(change.document_id)
                    continue
                asked.add((change.document_id, item.enricher.name))
                pending.append(item)
            if not pending:
                break
            report.rounds += 1
            by_client: dict[int, list[_BatchWork]] = {}
            for item in pending:
                by_client.setdefault(id(item.enricher.claude), []).append(item)
            for items in by_client.values():
                requests: list[tuple[str, dict[str, Any]]] = []
                for item in items:
                    for local, params in item.requests:
                        cid = f"r{len(requests):06d}"
                        item.ids[local] = cid
                        requests.append((cid, params))
                report.requests += len(requests)
                if progress:
                    progress(f"round {report.rounds}: {len(requests)} request(s)")
                answers = items[0].enricher.claude.batch(
                    requests, poll_seconds=poll_seconds, sleep=sleep, progress=progress
                )
                for item in items:
                    self._store_prefill(item, answers, ctx, report)
        return report

    def _next_model_call(
        self, doc: SourceDocument, ctx: StageContext, only: Sequence[str] | None
    ) -> _BatchWork | None:
        """The first model-backed enrichment this document still needs, as batch
        requests. Enrichers before it run as the build would run them."""
        parsed = with_document_facts(self._parse(doc, ctx))
        units = self._segment(parsed, ctx)
        enriched = [EnrichedUnit(unit=u) for u in units]
        prior: dict[Any, dict[str, Enrichment]] = {}
        for enricher in self.enrichers:
            keys = self._enrich_keys(enricher, parsed, enriched, prior)
            missing = [i for i, k in enumerate(keys) if ctx.cache.get(k) is None]
            batchable = callable(getattr(enricher, "requests_for", None)) and (
                only is None or enricher.name in only
            )
            if missing and batchable:
                first: dict[str, int] = {}
                for i in missing:
                    first.setdefault(keys[i], i)
                context = EnrichContext(
                    document=parsed, units=units, stage=ctx, prior=dict(prior), max_concurrency=1
                )
                reps = list(first.values())
                work = _BatchWork(enricher=enricher, context=context)
                for start in range(0, len(reps), self.enrich_batch_size):
                    chunk = reps[start : start + self.enrich_batch_size]
                    batch_units = [units[i] for i in chunk]
                    requests = enricher.requests_for(batch_units, context)  # type: ignore[attr-defined]
                    tag = f"b{start}"
                    work.groups.append(
                        (tag, batch_units, [keys[i] for i in chunk], [k for k, _ in requests])
                    )
                    work.requests.extend((f"{tag}:{k}", p) for k, p in requests)
                if work.requests:
                    return work
                continue
            if missing and not batchable and callable(getattr(enricher, "requests_for", None)):
                # A model-backed enricher left out of this prefill: what follows
                # may read its output, so the chain stops here for this document.
                return None
            results, _ = self._run_enricher(enricher, parsed, units, enriched, keys, prior, ctx)
            for i, e in results.items():
                enriched[i] = enriched[i].with_enrichment(e)
                prior.setdefault(enriched[i].unit_id, {})[e.enricher] = e
        return None

    def _store_prefill(
        self,
        work: _BatchWork,
        answers: Mapping[str, Any],
        ctx: StageContext,
        report: PrefillReport,
    ) -> None:
        enricher = work.enricher
        for tag, batch_units, keys, locals_ in work.groups:
            local_answers = {k: answers.get(work.ids.get(f"{tag}:{k}", "")) for k in locals_}
            for a in local_answers.values():
                report.cost_usd += float(getattr(a, "cost_usd", 0.0) or 0.0)
            made = enricher.enrichments_from(batch_units, work.context, local_answers)
            for key, e in zip(keys, made, strict=True):
                if e is None:
                    report.failed += 1
                    continue
                ctx.cache.put(key, json.dumps(encode_enrichment(e)).encode("utf-8"))
                report.cached += 1
                report.by_enricher[enricher.name] = report.by_enricher.get(enricher.name, 0) + 1


@dataclass(slots=True)
class _BatchWork:
    """One document's next model-backed enrichment, as batch requests."""

    enricher: Any
    context: EnrichContext
    #: (batch tag, units, their cache keys, the local request keys)
    groups: list[tuple[str, list[Unit], list[str], list[str]]] = field(default_factory=list)
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    #: local request key -> batch custom id
    ids: dict[str, str] = field(default_factory=dict)


def with_document_facts(parsed: ParsedDocument) -> ParsedDocument:
    """Add what the frame knows about every document to its metadata.

    Title, page count and language reach every unit -- and so every index as
    filterable fields, and the structured index's document record as the card
    an agent reads before opening a document. Existing keys win: a scanner's or
    parser's title is better than a guess.

    Only facts that an ordinary edit leaves alone: metadata is part of every
    unit's record, so a fact that changes with any edit (the text's hash, its
    length) would rewrite every unit of a document for a one-word change.
    """
    title = next((b.text for b in parsed.blocks if str(b.kind) == "heading"), "")
    facts = {
        "doc_title": title or parsed.metadata.get("title") or parsed.metadata.get("name"),
        "doc_pages": parsed.page_count,
        "doc_language": detect_language(parsed.text),
    }
    added = {k: v for k, v in facts.items() if v not in (None, "") and k not in parsed.metadata}
    return replace(parsed, metadata={**dict(parsed.metadata), **added}) if added else parsed


def metadata_hash(doc: SourceDocument) -> str:
    """The part of a document's scanner metadata that can change its indexing."""
    return hash_obj(content_metadata(doc.metadata))


def units_signature(units: Iterable[EnrichedUnit]) -> ContentHash:
    """Hash of a document's indexed surface. Used to detect no-op rebuilds."""
    return hash_text("\x00".join(u.indexing_text() for u in units))


# --------------------------------------------------------------------------- #
# parse identity                                                               #
# --------------------------------------------------------------------------- #


def parse_cache_key(parser: Parser, doc: SourceDocument) -> CacheKey:
    """The parse cache key: the bytes, plus whatever else the parser reads.

    Never the document's identity -- that is what lets byte-identical files
    share one parse -- but always the media type (or, for a routing parser, the
    parser it dispatches to). The same bytes named ``.md`` and ``.txt`` go to
    different parsers, and keying on bytes alone served the markdown parse to
    the text document.
    """
    return cache_key(
        parser.fingerprint(), doc.content_hash, scope_hash=parse_cache_scope(parser, doc)
    )


def strip_identity(parsed: ParsedDocument) -> ParsedDocument:
    """The parse with every trace of *which* document removed.

    What goes into the content-addressed cache. An entry shared by two tenants'
    copies of one file must not carry either tenant's id, URI or metadata.
    """
    return replace(
        parsed,
        document_id=DocumentId(""),
        source_uri="",
        metadata={},
        blocks=tuple(
            replace(b, provenance=replace(b.provenance, document_id=DocumentId(""), source_uri=""))
            for b in parsed.blocks
        ),
    )


def rebind_parsed(
    parsed: ParsedDocument, doc: SourceDocument, parser_metadata: Mapping[str, Any]
) -> ParsedDocument:
    """Stamp one document's identity onto a (possibly shared) parse.

    Scanner metadata wins over anything the parser added. The scanner states
    known facts -- the tenant a folder belongs to, the ACL a sidecar grants --
    while a parser reports what the bytes say, and a document must not be able
    to talk its way into another tenant by carrying a header of the same name.
    """
    return replace(
        parsed,
        document_id=doc.document_id,
        source_uri=doc.source_uri,
        source_hash=doc.content_hash,
        metadata={**dict(parser_metadata), **dict(doc.metadata)},
        blocks=tuple(
            replace(
                b,
                provenance=replace(
                    b.provenance, document_id=doc.document_id, source_uri=doc.source_uri
                ),
            )
            for b in parsed.blocks
        ),
    )


def cached_parse(parser: Parser, doc: SourceDocument, cache: CacheStore) -> ParsedDocument | None:
    """The parse a build left in the cache for ``doc``, identity stamped, or None.

    For tooling that needs parsed documents without re-parsing -- the golden-set
    bootstrapper reads them this way.
    """
    raw = cache.get(parse_cache_key(parser, doc))
    if raw is None:
        return None
    payload = json.loads(raw)
    return rebind_parsed(
        decode_parsed_document(payload), doc, decode_metadata(payload.get("parser_metadata", {}))
    )

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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from indexer.core.accounting import CacheOutcome, InMemoryAccountant
from indexer.core.cache import CacheStore, cache_key
from indexer.core.document import ParsedDocument, SourceDocument
from indexer.core.errors import ContractViolation, DocumentError
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
)
from indexer.core.unit import EnrichedUnit, Enrichment, Unit
from indexer.eval.checks import check_parsed_document, check_units
from indexer.pipeline.codec import (
    decode_parsed_document,
    decode_units,
    encode_parsed_document,
    encode_units,
)
from indexer.pipeline.stores import UnitStore

__all__ = ["BuildResult", "IngestionPipeline"]


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
        checkpoint_every: int = 200,
    ) -> None:
        self.scanner = scanner
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
            stale = [k for k, v in keys.items() if prior.stage_keys.get(k) != v]
            if stale:
                plan.append(
                    PlannedChange(
                        doc_id,
                        ChangeKind.RESTAGED,
                        tuple(stale),
                        f"stage fingerprint changed: {', '.join(sorted(stale))}",
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
        pending_records: list[DocumentRecord] = []

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
                self.ledger.delete(change.document_id)
                continue

            doc = by_id[change.document_id]
            if progress:
                progress(f"{change.kind.value:>9} {doc.source_uri}")
            try:
                parsed, _units, enriched = self._process(doc, ctx)
            except DocumentError as exc:
                failures.append(exc)
                stats.documents_failed += 1
                setattr(stats, counter, getattr(stats, counter) - 1)
                if self.on_document_error == "fail":
                    raise
                continue

            confidences.append(parsed.reading_order_confidence)
            stats.bytes_parsed += len(parsed.text)
            new_ids = [u.unit_id for u in enriched]
            prior_ids = list(change.prior.unit_ids) if change.prior else []
            to_upsert_ids, to_delete = diff_units(prior_ids, new_ids)

            # Restaging an index (a changed embedding model) must rewrite every
            # unit, not just the textually new ones -- the unit ids are the same
            # but their representation in that index is not.
            full_rewrite = change.kind is not ChangeKind.CHANGED or not prior_ids
            to_write = (
                enriched
                if full_rewrite
                else [u for u in enriched if u.unit_id in set(to_upsert_ids)]
            )

            if to_delete:
                stats.units_deleted += self._delete_units(list(to_delete), ctx)
            if to_write:
                self.unit_store.put_many(to_write)
                for i in range(0, len(to_write), self.index_batch_size):
                    batch = to_write[i : i + self.index_batch_size]
                    for name, idx in self.indexes.items():
                        with accountant.measure(idx.fingerprint()) as run:
                            receipt = idx.upsert(batch, ctx)
                            run.items_in = len(batch)
                            run.items_out = receipt.written
                            run.cost_usd = receipt.cost_usd
                            run.attrs = {"index": name}
                stats.units_written += len(to_write)
            stats.units_total += len(enriched)
            stats.units_reused_from_cache += len(enriched) - len(to_write)

            # Buffered, not committed. A ledger record claims a document is
            # done; committing it before the indexes holding that document are
            # durable is a lie the next build believes -- it skips the document
            # and the units are nowhere. That is not hypothetical: killing a
            # build mid-run left a ledger asserting 493 processed documents over
            # indexes that held none, and the resumed build skipped all 493.
            pending_records.append(
                DocumentRecord(
                    document_id=doc.document_id,
                    source_uri=doc.source_uri,
                    content_hash=doc.content_hash,
                    unit_ids=tuple(new_ids),
                    stage_keys=self.stage_keys(),
                    build_id=manifest.build_id,
                    updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
                )
            )
            if len(pending_records) >= self.checkpoint_every:
                self._checkpoint(pending_records)

        # Final checkpoint: everything still buffered becomes durable, and only
        # then are the remaining documents recorded as done.
        self._checkpoint(pending_records)

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
        return BuildResult(manifest=manifest, plan=work, failures=failures)

    # ------------------------------------------------------------- internals

    def _checkpoint(self, pending: list[DocumentRecord]) -> None:
        """Make index state durable, then record the documents it covers.

        The order is the whole point. Flushing after committing the ledger makes
        the ledger a claim about work that may not survive the process; flushing
        first makes every committed record backed by bytes on disk. A crash in
        between costs a redundant reprocess of the checkpoint's documents, which
        is the correct failure: too much work, never too little.
        """
        for idx in self.indexes.values():
            if isinstance(idx, Flushable):
                idx.flush()
        self.unit_store.flush()
        for record in pending:
            self.ledger.put(record)
        pending.clear()

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
    ) -> tuple[ParsedDocument, list[Unit], list[EnrichedUnit]]:
        parsed = self._parse(doc, ctx)
        units = self._segment(parsed, ctx)
        enriched = self._enrich(parsed, units, ctx)
        return parsed, units, enriched

    def _parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        fp = self.parser.fingerprint()
        key = cache_key(fp, doc.content_hash)
        cached = ctx.cache.get(key)
        if cached is not None:
            with ctx.accountant.measure(fp) as run:
                run.cache = CacheOutcome.HIT
                run.input_hash = doc.content_hash
                return decode_parsed_document(json.loads(cached))

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
            ctx.cache.put(key, json.dumps(encode_parsed_document(parsed)).encode("utf-8"))
        return parsed

    def _segment(self, parsed: ParsedDocument, ctx: StageContext) -> list[Unit]:
        fp = self.segmenter.fingerprint()
        key = cache_key(fp, parsed.content_hash)
        cached = ctx.cache.get(key)
        if cached is not None:
            with ctx.accountant.measure(fp) as run:
                run.cache = CacheOutcome.HIT
                return decode_units(json.loads(cached))

        with ctx.accountant.measure(fp) as run:
            run.cache = CacheOutcome.MISS
            run.input_hash = parsed.content_hash
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
        self, parsed: ParsedDocument, units: list[Unit], ctx: StageContext
    ) -> list[EnrichedUnit]:
        enriched = [EnrichedUnit(unit=u) for u in units]
        if not self.enrich_enabled:
            return enriched

        prior: dict[Any, dict[str, Enrichment]] = {}
        for enricher in self.enrichers:
            fp = enricher.fingerprint()
            # The scope decides what invalidates this enricher. A unit-scoped
            # enricher survives edits elsewhere in the document; a document-scoped
            # one does not. Including only what was read is what keeps an edit's
            # blast radius proportional to the edit.
            scope_hash = "" if str(enricher.scope) == "unit" else str(parsed.content_hash)
            todo: list[int] = []
            results: dict[int, Enrichment] = {}
            for i, eu in enumerate(enriched):
                key = cache_key(fp, eu.unit.content_hash, scope_hash=scope_hash)
                cached = ctx.cache.get(key)
                if cached is None:
                    todo.append(i)
                else:
                    from indexer.pipeline.codec import _dec_enrichment

                    results[i] = _dec_enrichment(json.loads(cached))

            if todo:
                batch = [enriched[i].unit for i in todo]
                with ctx.accountant.measure(fp) as run:
                    run.cache = CacheOutcome.MISS
                    run.items_in = len(batch)
                    try:
                        produced = enricher.enrich(
                            batch,
                            EnrichContext(
                                document=parsed, units=units, stage=ctx, prior=dict(prior)
                            ),
                        )
                    except Exception as exc:
                        raise DocumentError(
                            parsed.document_id, f"enrich:{fp.impl}", str(exc)
                        ) from exc
                    if len(produced) != len(batch):
                        raise ContractViolation(
                            f"{fp.key()}: returned {len(produced)} enrichments for "
                            f"{len(batch)} units; the contract is one per input, in order"
                        )
                    for i, e in zip(todo, produced, strict=True):
                        results[i] = e
                        from indexer.pipeline.codec import _enc_enrichment

                        ctx.cache.put(
                            cache_key(fp, enriched[i].unit.content_hash, scope_hash=scope_hash),
                            json.dumps(_enc_enrichment(e)).encode("utf-8"),
                        )
                    run.items_out = len(produced)
                    run.cost_usd = sum(e.cost_usd for e in produced)
                    run.tokens_in = sum(e.tokens_in for e in produced)
                    run.tokens_out = sum(e.tokens_out for e in produced)
            else:
                with ctx.accountant.measure(fp) as run:
                    run.cache = CacheOutcome.HIT
                    run.items_in = run.items_out = len(enriched)

            for i, e in results.items():
                enriched[i] = enriched[i].with_enrichment(e)
                prior.setdefault(enriched[i].unit_id, {})[e.enricher] = e
        return enriched


def units_signature(units: Iterable[EnrichedUnit]) -> ContentHash:
    """Hash of a document's indexed surface. Used to detect no-op rebuilds."""
    return hash_text("\x00".join(u.indexing_text() for u in units))

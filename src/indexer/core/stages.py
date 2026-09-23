"""The stage contracts.

    parse -> segment -> enrich -> index      (ingestion)
    route -> retrieve -> fuse -> rerank      (query)

Every stage is a ``Protocol`` with four things written down: its **input** and
**output** types, what it **may assume**, what it **must preserve**, and what
the **minimal implementation** is. The last one is a design test: a contract
whose minimal implementation is not obvious is over-specified, and one whose
minimal implementation is useless is under-specified.

Two rules hold across all eight:

**No stage imports another's internals.** A stage sees the previous stage's
output type and the frame's shared types. A segmenter that reaches into a
particular parser's ``attrs`` keys has coupled itself to that parser and the
swap this library exists to enable has been quietly given up.

**Every stage is purely a function of its declared inputs.** Not enforceable by
types, load-bearing for caching (``indexer.core.cache``), and the single most
likely contract to be broken by accident -- a clock read, a random seed, a
service that changed under you.

Disabling
---------
"Every stage individually disableable, so ablation is trivial" requires each
stage to have a defined pass-through, not just an on/off flag. Each protocol
below documents its own. Where the pass-through is not the identity function --
``parse`` and ``segment`` have no meaningful identity -- the disabled behaviour
is the *degenerate* implementation, which is itself the right baseline arm of an
ablation: parse-disabled means "decode the bytes", segment-disabled means "one
unit per document".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from indexer.core.accounting import Accountant, StageFingerprint
from indexer.core.cache import ArtifactStore, CacheStore
from indexer.core.document import ParsedDocument, SourceDocument, content_metadata
from indexer.core.ids import DocumentId, UnitId, hash_obj, merge_hashes
from indexer.core.predicate import Predicate, StructuredQuery
from indexer.core.query import Query, RouteDecision
from indexer.core.results import RankedList, RecordSet
from indexer.core.unit import ContextScope, EnrichedUnit, Enrichment, Unit

__all__ = [
    "CorpusScanner",
    "EnrichContext",
    "Enricher",
    "Flushable",
    "Fuser",
    "Index",
    "IndexKind",
    "IndexQuery",
    "IndexStatsView",
    "IndexWriteReceipt",
    "Parser",
    "Reranker",
    "Retriever",
    "Router",
    "Segmenter",
    "Stage",
    "StageContext",
    "StructuredCapable",
    "enrich_input_hash",
    "parse_cache_scope",
    "prior_hash",
    "unit_input_hash",
]


@dataclass(frozen=True, slots=True)
class StageContext:
    """Frame-supplied collaborators, passed to every stage call.

    Not configuration: none of this appears in a fingerprint, because none of it
    changes what a stage *should* produce -- only how fast it gets there. An
    implementation that lets the cache change its output has broken purity.
    """

    cache: CacheStore
    accountant: Accountant
    artifacts: ArtifactStore | None = None
    #: Cooperative cancellation for long builds. Implementations that loop over
    #: documents or batches should check it; the frame checks between stages.
    should_stop: Any = None
    attrs: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    """Common to all eight. ``fingerprint`` is what the manifest records."""

    def fingerprint(self) -> StageFingerprint: ...


# --------------------------------------------------------------------------- #
# corpus (pre-stage): finding documents                                        #
# --------------------------------------------------------------------------- #


@runtime_checkable
class CorpusScanner(Protocol):
    """Enumerates the corpus. Not one of the eight, but the incremental
    guarantee starts here.

    In / Out
        config -> iterable of ``SourceDocument``

    May assume
        Nothing about ordering or stability of the underlying store between
        scans.

    Must preserve
        A **stable ``document_id`` across scans** for the same logical document.
        If ids churn -- because they are derived from an absolute path that
        changed, or from content -- every rebuild looks like a full corpus
        replacement and incrementality silently stops working while appearing
        to function. This is the most commonly broken contract in the frame.

        ``content_hash`` must reflect the bytes that ``load()`` will return.

    Minimal implementation
        Walk a directory, id = hash of path relative to the root, content hash =
        sha256 of the file.
    """

    def scan(self) -> Iterable[SourceDocument]: ...

    def fingerprint(self) -> StageFingerprint: ...


# --------------------------------------------------------------------------- #
# parse                                                                        #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Parser(Protocol):
    """Bytes -> ordered typed blocks with provenance.

    In / Out
        ``SourceDocument`` -> ``ParsedDocument``

    May assume
        The bytes match the declared ``media_type``, and that it was selected
        for this document (by config or by a routing parser consulting
        ``can_parse``).

    Must preserve
        *Reading order*: ``blocks`` is the order a human reads in. A parser that
        cannot determine order must say so via ``reading_order_confidence``
        rather than emit stream order and let a segmenter build nonsense units
        from it.

        *Table structure*: a table block carries a ``Table`` payload. Rendering
        a table to a markdown string and discarding the grid destroys the
        structured path (invariant 5) at the first stage, and no later stage can
        recover it.

        *Span integrity*: ``parsed.text[b.span.start:b.span.end] == b.text`` for
        every block, with spans ascending and non-overlapping. This is checked,
        not trusted -- it is the root of every citation the system will ever
        emit.

        *Identity is the frame's.* Parse output is cached by content, so two
        byte-identical files -- the same PDF attached to fifty emails -- are
        parsed once. The frame then stamps each document's own id, URI and
        scanner metadata onto the result. A parser therefore must not derive
        its output from ``document_id``, ``source_uri`` or scanner metadata;
        anything it adds to ``metadata`` must come from the bytes. What else it
        reads (the media type, by default) it states in ``cache_scope``.

    Minimal implementation
        Decode UTF-8, split on blank lines, one PARAGRAPH block each, spans from
        the split offsets, confidence 1.0.

    Disabled
        ``PassthroughParser``: the whole document as one OTHER block. A valid
        ablation arm ("does structure-aware parsing earn its cost?"), not an
        error.

    Per-document routing
        Reference guidance is to classify and route per document rather than
        choose one parser globally -- a text-native PDF and a scan want
        different parsers, and a corpus contains both. ``can_parse`` returns a
        capability score in [0, 1]; a routing parser is itself a ``Parser`` that
        dispatches on it. This is why the frame has no ``parser_for_mimetype``
        table: the decision is per document and belongs to an implementation.
    """

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument: ...

    def can_parse(self, doc: SourceDocument) -> float:
        """Capability score in [0, 1]. 0 means "I cannot handle this at all"."""
        ...

    def fingerprint(self) -> StageFingerprint: ...


def parse_cache_scope(parser: Any, doc: SourceDocument) -> str:
    """What a parser reads besides the bytes. Part of the parse cache key.

    A parser may define ``cache_scope(doc) -> str``; a routing parser uses it to
    name the parser it would dispatch to, so the same bytes under ``.md`` and
    ``.txt`` are not served one cached parse. The default is the media type,
    which every parser is entitled to read.
    """
    fn = getattr(parser, "cache_scope", None)
    return str(fn(doc)) if callable(fn) else doc.media_type


# --------------------------------------------------------------------------- #
# segment                                                                      #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Segmenter(Protocol):
    """Parsed blocks -> units. Splits on structure, not character counts.

    In / Out
        ``ParsedDocument`` -> ``Sequence[Unit]``

    May assume
        Blocks are in reading order, typed, and span-consistent with
        ``parsed.text``. This is exactly why parse guarantees those things.

    Must preserve
        *Structure as the boundary signal.* Size parameters are **limits, not
        the primary criterion**. A segmenter that slices every 512 characters
        and consults ``BlockKind`` only to avoid mid-word breaks is not
        implementing this contract, whatever its parameter names say. The test:
        adding a paragraph to section 3 must not change the units of section 4.

        *Addressability.* Every unit carries ``document_id``, ``section_path``
        and a ``Provenance`` span that resolves in ``parsed.text``.

        *Table integrity.* A table is never split mid-row. A large table may be
        split by row groups, and then the header rows are repeated into each
        unit (``Table.header_rows`` exists for this) so every unit remains
        independently interpretable -- a chunk of table body with no column
        names is unretrievable and uninterpretable once retrieved.

        *Coverage.* Units cover the document's content blocks. Deliberate
        omissions (running headers, page numbers) are allowed and must be
        recorded in the returned units' absence, not silently merged.

    Minimal implementation
        One unit per leaf section; if a section exceeds the token limit, split
        at block boundaries, never inside a block.

    Disabled
        One unit per document. The honest baseline arm: it shows what chunking
        buys, and on small documents it sometimes wins.
    """

    def segment(self, parsed: ParsedDocument, ctx: StageContext) -> Sequence[Unit]: ...

    def fingerprint(self) -> StageFingerprint: ...


# --------------------------------------------------------------------------- #
# enrich                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EnrichContext:
    """What an enricher may read beyond the unit itself.

    The parent document is passed whole so that an implementation can exploit
    prompt caching over it -- the reference guidance is a small fast model with
    the parent document cached, so the document is not re-sent per chunk. That
    only works if the enricher receives units **batched by document**, which is
    why ``Enricher.enrich`` takes a sequence and this context is per batch.
    """

    document: ParsedDocument
    units: Sequence[Unit]
    stage: StageContext
    #: Enrichments already attached by earlier enrichers in the chain, so a
    #: classifier can read a summary. Creates an ordering dependency, which the
    #: config makes explicit: enrichers run in listed order.
    prior: Mapping[UnitId, Mapping[str, Enrichment]] = field(default_factory=dict)
    #: How many model calls an enricher may have in flight for this batch
    #: (``enrich.max_concurrency``). An enricher that makes no calls ignores it.
    max_concurrency: int = 1


@runtime_checkable
class Enricher(Protocol):
    """Per-unit augmentation before indexing. Where accuracy is won.

    In / Out
        ``Sequence[Unit]`` + ``EnrichContext`` -> ``Sequence[Enrichment]``,
        one per input unit, in the same order.

    May assume
        All units in a batch belong to ``ctx.document``, in reading order.

    Must preserve
        *Idempotence.* Running twice over the same inputs produces equal output.
        Not a nicety: the cache assumes it, and resumption after a crash
        depends on it.

        *Purity within the declared scope.* ``scope`` states what the enricher
        reads, and the cache key includes exactly that and nothing else. An
        enricher that declares ``UNIT`` and reads the document will serve stale
        results after a document edit -- a bug that survives cache clears and
        looks like a model regression.

        "The unit" means the whole unit, not its text: section path, kind and
        scanner metadata are part of it, and an enricher that copies a tenant
        out of ``unit.metadata`` reads the metadata. The frame's default key
        (``enrich_input_hash``) therefore covers all of it, plus the parent
        document for wider scopes and earlier enrichers' output unless the
        enricher sets ``reads_prior = False``. An enricher that reads less may
        say so precisely by defining ``input_hash(unit, document, prior)`` --
        narrowing is an optimisation, and getting it wrong is a stale cache, so
        the default errs wide.

        *Additivity.* An enricher writes its own key in ``EnrichedUnit
        .enrichments`` and never mutates another's. That is what makes
        enrichers individually disableable, which is what makes the
        contextualisation ablation a one-line config change.

    Minimal implementation
        A ``section_path`` prefixer: no model call, writes
        ``context = " > ".join(section_path)``. It is a genuinely useful
        baseline and it is the control arm that separates "contextualisation
        helps" from "any prefix helps" -- a distinction the sanity check in the
        eval harness would otherwise miss.

    Disabled
        No enrichments. ``indexing_text()`` degrades to the raw unit text, and
        every index follows automatically because none of them chooses its own
        retrieval surface.

    Why this is the expensive stage
        Invariant 3: a 50-100 token LLM-written summary prepended before *both*
        embedding and lexical indexing cuts top-20 retrieval failure from 5.7%
        to 2.9%. It is paid once per unit (invariant 2) and it is cached on
        content hash, so a re-run costs nothing.
    """

    name: str
    scope: ContextScope

    def enrich(self, units: Sequence[Unit], ctx: EnrichContext) -> Sequence[Enrichment]: ...

    def fingerprint(self) -> StageFingerprint: ...


def unit_input_hash(unit: Unit) -> str:
    """Everything a unit-scoped enricher may read of one unit.

    Text alone is not enough. A heading rename changes ``section_path`` and not
    the text, and a section-prefix context keyed on text kept serving the old
    heading. Two tenants holding the same paragraph differ only in metadata,
    and an extractor keyed on text copied the first tenant's id onto the
    second's unit -- which then answered the first tenant's filtered queries.
    """
    return hash_obj(
        {
            "text": unit.text,
            "section_path": list(unit.section_path),
            "kind": str(unit.kind),
            "table_ref": unit.table_ref,
            "metadata": content_metadata(unit.metadata),
        }
    )


def enrich_input_hash(
    enricher: Any,
    unit: Unit,
    document: ParsedDocument,
    prior: Mapping[str, Enrichment],
) -> str:
    """The cache input for one enricher on one unit.

    An enricher's own ``input_hash`` wins; otherwise the conservative default:
    the whole unit, the parent document for any scope wider than the unit, and
    the enrichments already attached unless the enricher declares
    ``reads_prior = False``. Third-party enrichers that say nothing get the
    widest key, because a needless cache miss costs a call and a missing input
    costs a wrong answer served from cache indefinitely.
    """
    custom = getattr(enricher, "input_hash", None)
    if callable(custom):
        return str(custom(unit, document, prior))
    parts = [unit_input_hash(unit)]
    if str(getattr(enricher, "scope", ContextScope.UNIT)) != ContextScope.UNIT:
        parts.append(str(document.content_hash))
        parts.append(hash_obj(content_metadata(document.metadata)))
    if getattr(enricher, "reads_prior", True) and prior:
        parts.append(prior_hash(prior))
    return merge_hashes(*parts)


def prior_hash(prior: Mapping[str, Enrichment]) -> str:
    """Hash of the enrichments earlier enrichers attached to one unit."""
    return hash_obj(
        {
            name: {
                "fingerprint": e.fingerprint,
                "context": e.context,
                "fields": dict(e.fields),
                "labels": {k: list(v) if isinstance(v, tuple) else v for k, v in e.labels.items()},
            }
            for name, e in sorted(prior.items())
        }
    )


# --------------------------------------------------------------------------- #
# index                                                                        #
# --------------------------------------------------------------------------- #

#: Open by design. ``IndexKind`` is a plain string, and **nothing in the frame
#: branches on its value**. That is the whole mechanism behind "adding a fourth
#: kind must not require touching the other three": there is no dispatch table
#: to extend, no enum to widen, no union to add a member to. Indexes are
#: configured by name, fanned out over uniformly, and asked the same question.
IndexKind = str

KIND_DENSE: IndexKind = "dense"
KIND_LEXICAL: IndexKind = "lexical"
KIND_STRUCTURED: IndexKind = "structured"
KIND_VISUAL: IndexKind = "visual"


@dataclass(frozen=True, slots=True)
class IndexQuery:
    """What an index is asked, in neutral form.

    Text, not vectors. A dense index embeds the text with the same model it
    indexed with; a lexical index analyses it; a visual index embeds it as a
    ColPali-style query. Handing an index a pre-computed vector would put the
    embedding model in the caller and couple every other index to that choice.
    """

    text: str
    top_k: int = 50
    filters: Predicate | None = None
    #: Restrict to specific units, for the iterative path's follow-up rounds
    #: and for reranking a candidate set.
    unit_ids: Sequence[UnitId] | None = None
    #: Implementation-specific knobs from the route decision (ef_search,
    #: analyzer choice). Uninterpreted by the frame.
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexWriteReceipt:
    written: int
    skipped: int = 0
    deleted: int = 0
    cost_usd: float = 0.0
    tokens: int = 0


@dataclass(frozen=True, slots=True)
class IndexStatsView:
    unit_count: int
    size_bytes: int | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Index(Protocol):
    """One index over the units. Several coexist over the same units.

    In / Out
        write: ``Sequence[EnrichedUnit]`` -> ``IndexWriteReceipt``
        read:  ``IndexQuery`` -> ``RankedList``

    May assume
        Units arrive enriched, with stable ids. Upserts may repeat: the same
        unit id may be written twice with identical content, and the index must
        treat that as a no-op rather than a duplicate (resumability depends on
        it).

    Must preserve
        *The shared retrieval surface.* An index MUST index
        ``unit.indexing_text()``. It may index more -- raw text in a second
        field, fields for filtering -- but the contextualised string is the
        surface. This is where invariant 3's "before both embedding and lexical
        indexing" is actually enforced; leaving it to each implementation is
        how the dense and lexical halves drift apart.

        *Provenance on every hit.* A hit carries ``Provenance``, so a store that
        holds only ids must persist the provenance payload alongside.

        *Deletion.* ``delete`` must actually remove. An index that tombstones
        without filtering at query time will keep answering with deleted
        documents, which is a correctness and often a compliance problem.

        *Dense ranks.* Hits are ranked 1..n with no gaps (``RankedList``
        enforces it), because fusion is rank-based.

    Minimal implementation
        Lexical: an in-memory BM25 over ``indexing_text()``. Dense: an in-memory
        matrix with exact cosine. Both are real, correct, and fast enough for a
        few thousand units -- which is the size of most golden sets.

    Disabled
        Removed from the fan-out. Neither written nor queried. Other indexes are
        unaffected, and the manifest records the omission.
    """

    name: str
    kind: IndexKind

    def upsert(self, units: Sequence[EnrichedUnit], ctx: StageContext) -> IndexWriteReceipt: ...

    def delete(self, unit_ids: Sequence[UnitId], ctx: StageContext) -> int: ...

    def delete_document(self, document_id: DocumentId, ctx: StageContext) -> int: ...

    def search(self, query: IndexQuery, ctx: StageContext) -> RankedList: ...

    def stats(self) -> IndexStatsView: ...

    def fingerprint(self) -> StageFingerprint: ...


@runtime_checkable
class Flushable(Protocol):
    """Capability: an index that batches durability and commits on demand.

    Persisting on every ``upsert`` makes ingestion quadratic for any index that
    rewrites a whole file or reindexes on commit -- a cost that is invisible on
    a ten-document test and fatal on a real corpus. So indexes are permitted to
    buffer, and the pipeline commits once per build.

    The contract: after ``flush`` returns, everything upserted is durable and
    searchable. Before it, results may be stale on disk but must already be
    correct in memory -- a query between batches must never see a partial
    index. An implementation with nothing to commit simply omits the capability.
    """

    def flush(self) -> None: ...


@runtime_checkable
class StructuredCapable(Protocol):
    """Capability protocol: an index that answers structured queries.

    A *capability*, not a subclass, and that is deliberate. Making ``Index``
    carry ``structured_query`` would force every dense store to implement or
    stub it, and each new capability would widen the interface every index must
    satisfy. Instead the frame asks ``isinstance(idx, StructuredCapable)``. A
    fifth kind with a capability of its own adds a protocol here and touches no
    existing index -- which is the stated requirement, generalised.
    """

    def structured_query(self, query: StructuredQuery, ctx: StageContext) -> RecordSet: ...


# --------------------------------------------------------------------------- #
# route                                                                        #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Router(Protocol):
    """Query -> dispatch decision. Where invariant 5 is enforced.

    In / Out
        ``Query`` -> ``RouteDecision``

    May assume
        The set of configured index names and which are ``StructuredCapable``,
        supplied at construction.

    Must preserve
        *At least three paths.* STRUCTURED goes straight to extracted fields and
        **never touches a vector index**. LOOKUP is exactly one retrieval pass.
        ITERATIVE loops under a step budget. The pipeline enforces the first two
        structurally: a STRUCTURED decision with dense targets is rejected, and
        LOOKUP with a budget above 1 is rejected at construction.

        *A budget on every decision.* No unbounded loops.

        *A logged decision, always.* Including when routing is disabled, when
        confidence is low, and when the decision was a fallback. The decision
        log is the only way to find out that 40% of traffic is taking the
        iterative path because a classifier drifted -- and routing errors are
        invisible in aggregate retrieval metrics, because the queries that were
        misrouted are precisely the ones whose gold the retriever never saw.

    Minimal implementation
        Rules over the query text: a regex for comparatives and superlatives, a
        date/number detector, a field-name lexicon built from the configured
        extraction schema. Cheap, ~0 latency, and a surprisingly strong baseline
        -- which is exactly why it is the control arm against an LLM router.

    Disabled
        Every query takes LOOKUP against all enabled indexes with the configured
        default ``top_k``, and a decision with ``reason="stage_disabled"`` is
        still logged. Note this deliberately violates invariant 5, and that is
        the point of the ablation: the harness will show structured queries
        failing, and the size of that gap is the router's measured value.
    """

    def route(self, query: Query, ctx: StageContext) -> RouteDecision: ...

    def fingerprint(self) -> StageFingerprint: ...


# --------------------------------------------------------------------------- #
# retrieve                                                                     #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Retriever(Protocol):
    """Executes the decision against the indexes. One list per index.

    In / Out
        ``Query`` + ``RouteDecision`` + indexes -> ``Sequence[RankedList]``

    May assume
        Targets name configured, enabled indexes. Filters on targets are already
        the conjunction of caller and inferred filters.

    Must preserve
        *One list per index*, unfused. Fusion is the next stage's job, and a
        retriever that pre-merges removes the ablation boundary between them.

        *The step budget.* The iterative loop lives **here**, not in a new
        stage: an implementation may issue follow-up rounds up to
        ``decision.step_budget``, recording each as a ``RetrievalStep``. Putting
        the loop in a stage of its own would mean the agentic path had a
        different pipeline shape from the simple path, and every downstream
        stage would need to know which it was in.

        *Isolation of failure.* One index erroring degrades to a shorter list
        with the error recorded; it does not fail the query. A reranker over
        three lists still works with two, and a query that returns something is
        better than a query that returns a stack trace.

    Minimal implementation
        Call ``search`` on each target sequentially and return the lists.

    Disabled
        Not disableable -- there is nothing to retrieve without it. Individual
        *indexes* are disableable, which is the ablation that matters
        ("dense-only", "lexical-only"), and parallel-vs-sequential execution is
        an implementation swap.
    """

    def retrieve(
        self,
        query: Query,
        decision: RouteDecision,
        indexes: Mapping[str, Index],
        ctx: StageContext,
    ) -> Sequence[RankedList]: ...

    def fingerprint(self) -> StageFingerprint: ...


# --------------------------------------------------------------------------- #
# fuse                                                                         #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Fuser(Protocol):
    """Several ranked lists -> one. Invariant 4's joint.

    In / Out
        ``Sequence[RankedList]`` -> ``RankedList``

    May assume
        Each input list is densely ranked from 1 and internally consistent.

    Must preserve
        *Score incomparability.* Scores from different indexes are not on a
        common scale, and a fuser that adds them is asserting a calibration it
        does not have. Reciprocal Rank Fusion is the default because it uses
        only position, which is the one thing the lists genuinely share. A
        score-normalising fuser is a legitimate alternative and must state its
        normalisation.

        *Deduplication by unit id*, keeping the best provenance and recording
        which indexes contributed -- the per-index contribution is what tells
        you whether the sparse half is earning its place.

        *Determinism.* Ties broken by a stated rule (unit id), so two runs of
        the same query produce the same order. Non-deterministic fusion makes
        every eval delta unreadable.

    Minimal implementation
        RRF: ``score(u) = sum_i weight_i / (k + rank_i(u))``, ``k = 60``.

    Disabled
        Concatenate in target order, dedup by unit id, keep first occurrence.
        For a single-index configuration this is exactly equivalent, which is
        what makes "is hybrid worth it?" a clean two-arm comparison.
    """

    def fuse(self, lists: Sequence[RankedList], ctx: StageContext) -> RankedList: ...

    def fingerprint(self) -> StageFingerprint: ...


# --------------------------------------------------------------------------- #
# rerank                                                                       #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Reranker(Protocol):
    """Reorders a candidate list with a stronger, slower model.

    In / Out
        ``Query`` + ``RankedList`` -> ``RankedList``

    May assume
        Candidates are fused and deduplicated, and each hit's ``matched_text``
        is populated (the reranker needs text, and a reranker that has to
        re-fetch it is being handed the wrong contract).

    Must preserve
        *Candidate set membership.* Reranking reorders and truncates; it never
        introduces a unit the retrievers did not return. A reranker that
        retrieves is a retriever, and hiding that in the rerank stage makes the
        latency budget unreadable.

        *A stated input width.* Reranking is O(candidates) in model calls, so
        ``input_top_k`` is a first-class parameter, not a hidden default: it is
        the main latency and cost dial on the query path.

    Minimal implementation
        Identity. Which is also the disabled behaviour, and that equivalence is
        deliberate -- "no reranker" and "the null reranker" must be the same
        arm, or the ablation table has two rows that should agree and might not.

    Reference guidance
        A cross-encoder (the BGE reranker family). Invariant 3's numbers put
        reranking at 2.9% -> 1.9% top-20 failure after contextualisation.
        **Avoid LLM-as-reranker on the simple path**: the majority of traffic
        takes LOOKUP, and an LLM there spends the latency budget on the queries
        that least need it. The seam is open (a reranker is just a reranker) but
        the config's default binds an LLM reranker to the ITERATIVE path only.
    """

    def rerank(self, query: Query, candidates: RankedList, ctx: StageContext) -> RankedList: ...

    def fingerprint(self) -> StageFingerprint: ...

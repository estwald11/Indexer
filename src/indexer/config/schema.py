"""The config schema. A project is one of these files.

Two-phase validation
--------------------
The schema is deliberately *not* a closed enumeration of every implementation
and its parameters. If it were, adding an implementation would mean editing the
schema, and "a new project is a config file, not a fork" would be false for
exactly the case that matters.

So validation happens twice:

**Phase 1 (here, pydantic).** Shape, types, cross-stage coherence. Every
implementation is named by a string and its ``params`` is an open mapping.

**Phase 2 (``loader.resolve``).** Each name is looked up in the registry and its
``params`` validated against the params model that implementation registered.

The result is that a typo in an implementation-specific parameter is still a
startup error naming the field -- the usual reason people give up on open
schemas -- while the frame keeps no list of implementations.

Enabled flags
-------------
``enabled: false`` appears on every stage, and each stage's disabled behaviour
is defined in ``indexer.core.stages`` (identity where that is meaningful; the
degenerate implementation where it is not). This is what makes ablation a config
edit rather than a code path.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AblationSpec",
    "AccessConfig",
    "AccountingConfig",
    "CacheConfig",
    "Config",
    "CorpusConfig",
    "EnrichConfig",
    "EvalConfig",
    "FuseConfig",
    "ImplSpec",
    "IndexConfig",
    "IndexSpec",
    "IngestionConfig",
    "ParseConfig",
    "QueryConfig",
    "RerankConfig",
    "RetrieveConfig",
    "RouteConfig",
    "SegmentConfig",
    "ShapeConfig",
    "SourceSpec",
]

SCHEMA_VERSION = 1


class _Base(BaseModel):
    # Unknown keys are errors, not silently ignored. A misspelled `top_k:` that
    # is accepted and does nothing is the worst kind of config bug: the system
    # works, just not as configured, and nothing anywhere says so.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImplSpec(_Base):
    """A named implementation plus its parameters. The frame's only plug shape."""

    impl: str
    enabled: bool = True
    #: Validated in phase 2 against the registered params model.
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# corpus                                                                       #
# --------------------------------------------------------------------------- #


class SourceSpec(ImplSpec):
    """Where documents come from. Multiple sources feed one index."""

    #: Prefix for derived document ids, so two sources cannot collide on a
    #: shared relative path -- which would silently overwrite one with the other.
    namespace: str = ""


class CorpusConfig(_Base):
    sources: list[SourceSpec] = Field(min_length=1)
    #: Cap for smoke runs and eval bootstrapping. ``None`` means the whole corpus.
    limit: int | None = None


# --------------------------------------------------------------------------- #
# ingestion stages                                                             #
# --------------------------------------------------------------------------- #


class ParseRoute(_Base):
    """Per-document parser selection.

    Reference guidance is to classify and route per document rather than pick
    one parser globally: a text-native PDF and a 300-page scan want different
    parsers and a real corpus holds both. ``when`` is matched against the
    ``SourceDocument`` (media type, extension, size, and any classifier label);
    the first match wins, and ``ParseConfig.default`` catches the rest.
    """

    when: dict[str, Any]
    impl: str
    params: dict[str, Any] = Field(default_factory=dict)


class ParseConfig(_Base):
    enabled: bool = True
    #: Used when no route matches. Also the whole story for a single-format corpus.
    default: ImplSpec
    routes: list[ParseRoute] = Field(default_factory=list)
    #: Fail the build, or record the document as failed and continue. At corpus
    #: scale the second is the only usable answer; the count lands in the manifest.
    on_error: Literal["fail", "skip"] = "skip"
    #: Below this, the document is flagged in the manifest. Not an error: a
    #: warning that the corpus has outgrown its parser.
    min_reading_order_confidence: float = 0.5


class SegmentConfig(ImplSpec):
    """Disabled behaviour: one unit per document."""

    #: A *limit*, not the splitting criterion. The contract requires boundaries
    #: to come from structure; this caps the result.
    max_tokens: int = 512
    min_tokens: int = 64
    #: Overlap is a chunking-era workaround for lost context. Invariant 3 says
    #: the fix is contextualisation, not overlap, so this defaults to 0 and is
    #: kept only as an ablation arm -- "does overlap add anything once units are
    #: contextualised?" is a question worth being able to answer with a number.
    overlap_tokens: int = 0


class EnricherSpec(ImplSpec):
    """One enricher in the chain. Order is significant: later ones read earlier
    output via ``EnrichContext.prior``."""

    #: What this enricher reads, hence what invalidates its cache. Must match
    #: the implementation's declared scope; the loader checks and refuses a
    #: config that claims a narrower scope than the code takes, because that
    #: combination serves stale results after an edit.
    scope: Literal["unit", "neighbors", "document", "corpus"] | None = None


class EnrichConfig(_Base):
    enabled: bool = True
    enrichers: list[EnricherSpec] = Field(default_factory=list)
    #: Units per model call. Batching by document is what makes prompt caching
    #: over the parent document pay off -- the document is sent once, not once
    #: per chunk.
    batch_size: int = 16
    max_concurrency: int = 4
    #: A failed enrichment leaves the unit un-enriched and indexable, or fails
    #: the document. Default keeps the corpus complete at slightly lower quality.
    on_error: Literal["fail", "skip"] = "skip"


class IndexSpec(ImplSpec):
    """One index. Adding a fourth kind is adding an entry here."""

    name: str
    #: Free string. Nothing in the frame branches on it; it is for the manifest,
    #: for routing targets, and for humans reading an ablation table.
    kind: str


class IndexConfig(_Base):
    indexes: list[IndexSpec] = Field(min_length=1)
    #: Units per write batch, across all indexes.
    batch_size: int = 128

    @model_validator(mode="after")
    def _unique_names(self) -> IndexConfig:
        names = [i.name for i in self.indexes]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate index names: {sorted(dupes)}")
        return self


class IngestionConfig(_Base):
    parse: ParseConfig
    segment: SegmentConfig
    enrich: EnrichConfig = Field(default_factory=lambda: EnrichConfig())
    index: IndexConfig
    #: Documents between durability checkpoints, and therefore the unit of
    #: resumability: an interrupted build resumes at the last checkpoint, not at
    #: the last document. Lower costs more flushes; higher risks more rework
    #: after a crash. It is not "how often to save" -- a ledger record is only
    #: written once the indexes holding that document are durable, so this is
    #: exactly how much work a crash can cost.
    checkpoint_every: Annotated[int, Field(ge=1)] = 200


# --------------------------------------------------------------------------- #
# query stages                                                                 #
# --------------------------------------------------------------------------- #


class PathSpec(_Base):
    """Per-route-path overrides. How LOOKUP and ITERATIVE differ in config."""

    targets: list[str] = Field(default_factory=list)
    top_k: dict[str, int] = Field(default_factory=dict)
    step_budget: int = 1
    #: Per-path reranker override. This is where "no LLM reranker on the simple
    #: path" is expressed: LOOKUP names the cross-encoder, ITERATIVE may name
    #: something slower, because it is already a slow path and a small share of
    #: traffic.
    rerank: str | None = None


class RouteConfig(_Base):
    """Disabled behaviour: everything takes LOOKUP against all indexes, and the
    decision is still logged with ``reason="stage_disabled"``."""

    enabled: bool = True
    impl: str = "rules"
    params: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, PathSpec] = Field(default_factory=dict)
    #: Where decisions are written. Invariant: every decision is logged.
    #:
    #: ``None`` means the assembler's default, ``<paths.store>/route-decisions
    #: .jsonl``. It deliberately does *not* default to an interpolated string:
    #: interpolation runs over the config file before validation, so a token in
    #: a schema default is never expanded and would be taken literally -- which
    #: creates a directory named "${paths.store}".
    decision_log: str | None = None
    #: Set false to write no decision log. Distinct from leaving the path unset.
    log_decisions: bool = True
    #: Below this confidence, fall back to this path rather than commit to a
    #: guess. Misrouting is expensive precisely because it is invisible in
    #: aggregate retrieval metrics.
    min_confidence: float = 0.0
    fallback_path: Literal["structured", "lookup", "iterative"] = "lookup"

    @model_validator(mode="after")
    def _three_paths(self) -> RouteConfig:
        if not self.enabled:
            return self
        required = {"structured", "lookup", "iterative"}
        missing = required - set(self.paths)
        if missing:
            raise ValueError(
                f"route requires at least the three paths {sorted(required)}; "
                f"missing {sorted(missing)}. Structured queries reaching vector "
                f"search is the failure this stage exists to prevent."
            )
        if self.paths["lookup"].step_budget != 1:
            raise ValueError("the lookup path is one retrieval pass by definition")
        return self


class RetrieveConfig(ImplSpec):
    impl: str = "parallel"
    #: Default depth per index, overridable per path and per decision.
    default_top_k: int = 50
    timeout_ms: int = 5000
    #: One index failing yields a shorter candidate set rather than a failed
    #: query. The error is recorded on the response either way.
    on_index_error: Literal["fail", "degrade"] = "degrade"


class FuseConfig(ImplSpec):
    """Disabled behaviour: concatenate in target order, dedup, keep first."""

    impl: str = "rrf"
    enabled: bool = True
    #: RRF's rank constant. 60 is the published default and a reasonable prior;
    #: it is exposed because it is worth an ablation on a new corpus.
    k: int = 60
    #: Per-index weights. Absent means 1.0. Invariant 4 says hybrid beats either
    #: half; the weights are how you find out by how much, on this corpus.
    weights: dict[str, float] = Field(default_factory=dict)
    #: Weights per query type, overriding ``weights`` for questions of that
    #: type. ``indexer.eval.tuning`` fits them on a development split.
    weights_by_type: dict[str, dict[str, float]] = Field(default_factory=dict)


class RerankConfig(ImplSpec):
    """Disabled behaviour: identity -- the same arm as ``impl: noop``."""

    impl: str = "noop"
    enabled: bool = True
    #: Candidates fed to the reranker. The main cost and latency dial on the
    #: query path, so it is explicit rather than an implementation default.
    input_top_k: int = 50
    output_top_k: int = 10


class AccessConfig(_Base):
    """Document-level access control, enforced on every path.

    Off by default, because the reference corpora have no ACLs. On, a query
    must state its principals, and sees only documents whose ``field`` names
    one of them -- in every index, the structured one included, since the
    check is a filter conjoined to the caller's own.
    """

    enabled: bool = False
    #: The metadata field holding each document's ACL (a list of principals).
    field: str = "acl"
    #: What a document with no ACL means. Deny unless the archive is known to
    #: be open by default: an omitted ACL is far more often a missing sidecar
    #: than a decision to publish.
    missing: Literal["deny", "allow"] = "deny"


class ShapeConfig(_Base):
    """What the final result list looks like to whoever reads it -- usually an agent.

    Off by default to keep the published ablation's arms unchanged.
    """

    enabled: bool = False
    #: At most this many units per document in the final list; 0 means no
    #: limit. Five chunks of one manual crowd out the other four documents.
    max_per_document: int = 0
    #: Collapse hits from documents with identical text -- the same contract
    #: saved in three folders -- into the best-ranked one, listing the others.
    collapse_duplicates: bool = True
    #: Also collapse documents whose text SimHashes are within this many bits.
    #: 0 disables it, and that is the default: two invoices from one template
    #: differ in a few words and are *not* duplicates.
    near_duplicate_bits: int = 0
    #: Attach the text of this many units before and after each hit, so a
    #: reader gets the passage around a chunk without another call.
    expand_neighbors: int = 0


class QueryConfig(_Base):
    route: RouteConfig
    retrieve: RetrieveConfig = Field(default_factory=lambda: RetrieveConfig())
    fuse: FuseConfig = Field(default_factory=lambda: FuseConfig())
    rerank: RerankConfig = Field(default_factory=lambda: RerankConfig())
    access: AccessConfig = Field(default_factory=lambda: AccessConfig())
    shape: ShapeConfig = Field(default_factory=lambda: ShapeConfig())


# --------------------------------------------------------------------------- #
# cross-cutting                                                                #
# --------------------------------------------------------------------------- #


class CacheConfig(ImplSpec):
    impl: str = "filesystem"
    enabled: bool = True
    #: Per-stage override, for the case where one stage's cache must be dropped
    #: (a prompt change that was not version-bumped) without paying to re-parse
    #: the corpus.
    stages: dict[str, bool] = Field(default_factory=dict)
    #: Delete cache entries no current document uses -- those of removed
    #: documents and of the previous version of edited ones -- at the end of
    #: every build. On by default: the cache holds full text, LLM summaries
    #: and extracted fields, and a removed document must not survive in it.
    #: Off only trades that for cheaper reverts of edits.
    purge_unreferenced: bool = True


class AccountingConfig(_Base):
    enabled: bool = True
    #: USD per 1M tokens, per model id. Kept in config rather than code because
    #: prices change and a stale constant silently misreports cost per query --
    #: one of the six required metrics.
    prices: dict[str, dict[str, float]] = Field(default_factory=dict)
    sink: str | None = None


class PathsConfig(_Base):
    store: str = "./var/index"
    cache: str = "./var/cache"
    manifests: str = "./var/manifests"
    artifacts: str = "./var/artifacts"


# --------------------------------------------------------------------------- #
# eval                                                                         #
# --------------------------------------------------------------------------- #


class AblationSpec(_Base):
    """One arm of an ablation. Overrides are dotted config paths.

    Expressing an arm as an override list rather than a second config file
    matters: the arms then provably differ in exactly the stated keys, and the
    delta table can label each row with the change that produced it.
    """

    name: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class SanityCheck(_Base):
    """A published expectation, asserted against this corpus.

    The brief states two: contextualisation should cut retrieval failures by
    roughly a third, and reranking should roughly halve them again. Encoding
    them as bounds turns "something is wired wrong, investigate" into a harness
    output rather than a thing someone might notice.
    """

    name: str
    metric: str = "retrieval_failure_rate"
    baseline_arm: str
    treatment_arm: str
    #: Expected relative reduction, as a fraction. 0.33 means "a third better".
    expected_reduction: float
    tolerance: float = 0.5


class EvalConfig(_Base):
    golden_set: str = "./eval/golden.jsonl"
    #: Depths for recall@k and nDCG@k. 5 is always included: Precision@5 is the
    #: metric that predicts answer accuracy at r=0.98, so it is not optional.
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    #: Top-k for the retrieval failure rate. 20 matches the published numbers
    #: the sanity checks are calibrated against.
    failure_k: int = 20
    #: How a hit is judged relevant against gold. Gold is anchored to document
    #: spans, not unit ids, so that one golden set survives re-segmentation --
    #: without which no two ablation arms would be comparable.
    match: Literal["span_overlap", "span_containment", "unit_id"] = "span_overlap"
    min_overlap: float = 0.5
    judge: ImplSpec | None = None
    bootstrap: ImplSpec | None = None
    ablations: list[AblationSpec] = Field(default_factory=list)
    sanity_checks: list[SanityCheck] = Field(default_factory=list)
    report_dir: str = "./var/eval"


# --------------------------------------------------------------------------- #
# root                                                                         #
# --------------------------------------------------------------------------- #


class ProjectConfig(_Base):
    name: str
    description: str = ""


class Config(_Base):
    """The root. One file per project, committable, secret-free."""

    schema_version: Annotated[int, Field(ge=1, le=SCHEMA_VERSION)] = SCHEMA_VERSION
    project: ProjectConfig
    paths: PathsConfig = Field(default_factory=lambda: PathsConfig())
    corpus: CorpusConfig
    ingestion: IngestionConfig
    query: QueryConfig
    cache: CacheConfig = Field(default_factory=lambda: CacheConfig())
    accounting: AccountingConfig = Field(default_factory=lambda: AccountingConfig())
    eval: EvalConfig = Field(default_factory=lambda: EvalConfig())
    #: Path to a base config this one overlays. Resolved before validation.
    extends: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Config:
        """Cross-stage checks that no single stage can make for itself."""
        index_names = {i.name for i in self.ingestion.index.indexes if i.enabled}
        if not index_names:
            raise ValueError("every index is disabled; there would be nothing to retrieve from")

        # Invariant 5 is a property of the *configuration*, not just the router:
        # a structured path with no structured index cannot avoid vector search,
        # so the config that promises it is rejected here rather than degrading
        # silently at query time.
        if self.query.route.enabled:
            structured_kinds = {
                i.name for i in self.ingestion.index.indexes if i.enabled and i.kind == "structured"
            }
            path = self.query.route.paths.get("structured")
            if path is not None and path.targets:
                unknown = set(path.targets) - index_names
                if unknown:
                    raise ValueError(
                        f"route.paths.structured targets unknown or disabled indexes: "
                        f"{sorted(unknown)}"
                    )
                if not (set(path.targets) & structured_kinds):
                    raise ValueError(
                        "route.paths.structured targets no index of kind 'structured'. "
                        "Structured, numeric and temporal questions would reach vector "
                        "search -- configure a structured index or disable the path."
                    )

            for name, spec in self.query.route.paths.items():
                unknown = set(spec.targets) - index_names
                if unknown:
                    raise ValueError(
                        f"route.paths.{name} targets unknown or disabled indexes: {sorted(unknown)}"
                    )

        # A weight naming an index that does not exist is a typo and does
        # nothing -- an error. A weight naming a *disabled* index is an ablation
        # arm mid-flight, where the weight is simply inert. Conflating the two
        # makes every "turn this index off" arm unexpressible.
        all_index_names = {i.name for i in self.ingestion.index.indexes}
        misspelled = set(self.query.fuse.weights) - all_index_names
        if misspelled:
            raise ValueError(f"fuse.weights names indexes that do not exist: {sorted(misspelled)}")

        return self

    @property
    def enrich_enabled(self) -> bool:
        return self.ingestion.enrich.enabled and any(
            e.enabled for e in self.ingestion.enrich.enrichers
        )

    def extracted_field_names(self) -> list[str]:
        """Fields the enabled enrichers declare. This is the router's vocabulary.

        Read from config rather than from a built index, so a misconfiguration
        is catchable before a build rather than after one.
        """
        names: list[str] = []
        if not self.ingestion.enrich.enabled:
            return names
        for spec in self.ingestion.enrich.enrichers:
            if not spec.enabled:
                continue
            names.extend(spec.params.get("fields", {}) or {})
            names.extend(spec.params.get("from_metadata", []) or [])
            schema = spec.params.get("schema")
            if isinstance(schema, dict):
                names.extend(schema)
        return sorted(set(names))

    def extracted_field_types(self) -> dict[str, str]:
        """Declared type per extracted field: str, int, float, bool or date.

        The router needs these. Without them it must guess a value's type from
        how it is written, and "version 1.0.0" reads as the float 1.0 -- which
        queries a numeric column while the value lives in a text one, so a
        well-formed question matches nothing. The types are already stated in
        the extraction config; the only bug was not passing them along.
        """
        out: dict[str, str] = {}
        if not self.ingestion.enrich.enabled:
            return out
        for spec in self.ingestion.enrich.enrichers:
            if not spec.enabled:
                continue
            for name, decl in (spec.params.get("fields", {}) or {}).items():
                if isinstance(decl, dict):
                    out[name] = str(decl.get("type", "str"))
            schema = spec.params.get("schema")
            if isinstance(schema, dict):
                for name, t in schema.items():
                    out[name] = str(t)
            for name in spec.params.get("from_metadata", []) or []:
                out.setdefault(name, "str")
        return out

    def warnings(self) -> list[str]:
        """Configurations that are valid but probably not what was meant.

        Warnings rather than errors because each is a legitimate ablation arm.
        They are printed at load and recorded in the manifest, so a production
        index built from an ablation config is identifiable after the fact.
        """
        out: list[str] = []
        if not self.enrich_enabled:
            out.append(
                "enrich is disabled: units are indexed without contextualisation. "
                "Published numbers put top-20 retrieval failure at 5.7% without it "
                "versus 2.9% with. Intended only as an ablation arm."
            )
        kinds = {i.kind for i in self.ingestion.index.indexes if i.enabled}
        if not {"dense", "lexical"} <= kinds:
            out.append(
                f"indexes cover {sorted(kinds)}: hybrid retrieval needs both a dense "
                f"and a lexical index. Either half alone is measurably worse."
            )
        if not self.query.rerank.enabled or self.query.rerank.impl == "noop":
            out.append(
                "rerank is off: published numbers put it at 2.9% -> 1.9% top-20 "
                "retrieval failure once units are contextualised."
            )
        # A structured index with nothing in it is the quietest way to lose
        # invariant 5. The config validates -- a structured index exists and the
        # route targets it -- but no enricher extracts a field, so the router's
        # lexicon is empty, every structured question is classified as prose, and
        # the structured path never fires. Nothing errors; the questions just
        # fail. This is a warning rather than an error because an index being
        # populated later, or by a caller's own enricher, is legitimate.
        if self.query.route.enabled and "structured" in self.query.route.paths:
            structured_live = any(
                i.kind == "structured" and i.enabled for i in self.ingestion.index.indexes
            )
            if structured_live and not self.extracted_field_names():
                out.append(
                    "a structured index and a structured route are configured, but no "
                    "enabled enricher declares any field to extract. The router has no "
                    "field vocabulary, so structured and numeric questions will be "
                    "classified as prose and sent to vector search -- the exact failure "
                    "invariant 5 exists to prevent. Add an extraction enricher, or "
                    "remove the structured path."
                )

        enabled_names = {i.name for i in self.ingestion.index.indexes if i.enabled}
        inert = set(self.query.fuse.weights) - enabled_names
        if inert and self.query.fuse.enabled:
            out.append(
                f"fuse.weights sets weights for disabled indexes {sorted(inert)}; "
                f"they have no effect."
            )
        if 5 not in self.eval.k_values:
            out.append(
                "eval.k_values omits 5; Precision@5 is the metric that tracks answer accuracy."
            )
        return out

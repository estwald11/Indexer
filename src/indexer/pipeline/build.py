"""Config -> running pipeline. Phase-2 validation happens here.

This is the only place that knows both the config schema and the registry, and
keeping it in one file is deliberate: it is the seam where "a new project is a
config file" is either true or false, and a reviewer should be able to check
that claim by reading one module.

Nothing here names an implementation. Every `impl:` string goes through the
registry, and the params it carries are validated against whatever that
implementation declared.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import indexer.impls  # noqa: F401  -- registers the reference implementations
from indexer.config.loader import config_hash as _config_hash
from indexer.config.loader import load, redact
from indexer.config.schema import Config
from indexer.core.cache import CacheStore, NullCache
from indexer.core.errors import ConfigError
from indexer.core.registry import Registry, resolve
from indexer.core.stages import Index
from indexer.pipeline.ingest import IngestionPipeline
from indexer.pipeline.query import AccessPolicy, QueryEngine, ShapePolicy
from indexer.pipeline.stores import CacheRefs, FileArtifactStore, FileCache, JsonLedger, UnitStore

__all__ = ["Assembly", "assemble", "assemble_mapping", "build_indexes", "declared_fields"]


class Assembly:
    """Everything a config describes, constructed.

    Ingestion and query are built together from one config so they cannot
    disagree about which indexes exist -- a query engine pointed at an index the
    ingestion pipeline never wrote is a class of bug worth making impossible.
    """

    def __init__(
        self,
        config: Config,
        resolved: Mapping[str, Any],
        *,
        registry: Registry | None = None,
        llm_client: Any = None,
    ) -> None:
        self.config = config
        #: The client every model-backed stage calls through, when the caller
        #: supplies one: a Bedrock, Vertex or Foundry client, a proxy, a fake
        #: in tests. None builds ``anthropic.Anthropic()`` on first use.
        self.llm_client = llm_client
        self.resolved = dict(resolved)
        self.config_hash = _config_hash(resolved)
        self.registry = registry
        self.paths = config.paths

        for d in (self.paths.store, self.paths.cache, self.paths.manifests, self.paths.artifacts):
            Path(d).mkdir(parents=True, exist_ok=True)

        self.cache: CacheStore = (
            FileCache(self.paths.cache) if config.cache.enabled else NullCache()
        )
        self.artifacts = FileArtifactStore(self.paths.artifacts)
        self.ledger = JsonLedger(Path(self.paths.store) / "ledger.json")
        self.unit_store = UnitStore(Path(self.paths.store) / "units.json")
        self.cache_refs = CacheRefs(Path(self.paths.store) / "cache-refs.db")
        self.indexes: dict[str, Index] = build_indexes(config, self.paths.store, registry)

    # --------------------------------------------------------------- stages

    def _build(self, stage: str, impl: str, params: Mapping[str, Any], **extra: Any) -> Any:
        reg = resolve(stage, impl, self.registry)
        return reg.build(params, **extra)

    def scanner(self) -> Any:
        """Every source, built through the registry like any other stage.

        It used to construct ``FilesystemScanner`` directly whatever ``impl``
        said, so a registered scanner for a DMS or a mailbox could be named in
        config and never run. And ``corpus.limit`` applied only when there were
        several sources; with one, a smoke run over "the first 50 documents"
        read the whole archive.
        """
        sources = self.config.corpus.sources
        state_dir = Path(self.paths.store) / "scan-state"
        built = [
            resolve("corpus", s.impl, self.registry).build(
                s.params, namespace=s.namespace, state_dir=state_dir
            )
            for s in sources
        ]
        if len(built) == 1 and self.config.corpus.limit is None:
            return built[0]
        return _MultiScanner(built, limit=self.config.corpus.limit)

    def parser(self) -> Any:
        p = self.config.ingestion.parse
        if not p.enabled:
            return self._build("parse", "passthrough", {})
        if p.routes:
            return self._build(
                "parse",
                "routing",
                {
                    "routes": [
                        {"when": r.when, "impl": r.impl, "params": r.params} for r in p.routes
                    ],
                    "default": {"impl": p.default.impl, "params": p.default.params},
                },
            )
        return self._build("parse", p.default.impl, p.default.params)

    def segmenter(self) -> Any:
        s = self.config.ingestion.segment
        if not s.enabled:
            return self._build("segment", "whole_document", {})
        # Size limits live on the stage config rather than inside params so they
        # are visible in every config at the same place; they are merged into the
        # implementation's params here.
        params = dict(s.params)
        params.setdefault("max_tokens", s.max_tokens)
        params.setdefault("min_tokens", s.min_tokens)
        params.setdefault("overlap_tokens", s.overlap_tokens)
        reg = resolve("segment", s.impl, self.registry)
        accepted = _accepted_params(reg, params)
        return reg.build(accepted)

    def enrichers(self) -> list[Any]:
        e = self.config.ingestion.enrich
        if not e.enabled:
            return []
        out = []
        for spec in e.enrichers:
            if not spec.enabled:
                continue
            # Prices are frame-supplied, like the cache: configured once, for
            # every model-backed stage, and never part of a fingerprint -- a
            # price change must not invalidate an enrichment.
            impl = self._build(
                "enrich",
                spec.impl,
                spec.params,
                prices=self.config.accounting.prices,
                client=self.llm_client,
            )
            # An enricher whose declared scope is narrower than the code's would
            # serve stale results after an edit -- a bug that survives cache
            # clears. Config cannot widen what the implementation declares.
            if spec.scope is not None and str(impl.scope) != spec.scope:
                raise ConfigError(
                    f"enricher {spec.impl!r} declares scope {impl.scope!r} but config "
                    f"says {spec.scope!r}. The cache key follows the declared scope, so "
                    f"the narrower value would serve stale enrichments after an edit."
                )
            out.append(impl)
        return out

    def router(self) -> Any:
        r = self.config.query.route
        if not r.enabled:
            return self._build(
                "route",
                "passthrough",
                {
                    "targets": list(self.indexes),
                    "top_k": self.config.query.retrieve.default_top_k,
                },
            )
        params = dict(r.params)
        params.setdefault(
            "paths",
            {
                name: {
                    "targets": spec.targets or list(self.indexes),
                    "top_k": spec.top_k,
                    "step_budget": spec.step_budget,
                }
                for name, spec in r.paths.items()
            },
        )
        params.setdefault("default_top_k", self.config.query.retrieve.default_top_k)
        # The router learns the corpus's field vocabulary from what the
        # configured stages declare they write, so a new corpus teaches it
        # without a code change.
        names, types = declared_fields(self.config, self.registry)
        params.setdefault("field_lexicon", names)
        params.setdefault("field_types", types)
        params.setdefault(
            "enable_structured",
            any(i.kind == "structured" and i.enabled for i in self.config.ingestion.index.indexes),
        )
        reg = resolve("route", r.impl, self.registry)
        # A router that reads the structured index's own description -- types,
        # ranges, frequent values -- gets it from here. Frame-supplied: the
        # description changes with every build and is no part of a fingerprint.
        structured = next(
            (i for i in self.indexes.values() if callable(getattr(i, "describe_schema", None))),
            None,
        )
        schema = getattr(structured, "describe_schema", None)
        return reg.build(_accepted_params(reg, params), schema=schema, client=self.llm_client)

    def _field_lexicon(self) -> list[str]:
        """The router's field vocabulary. One derivation, shared with the
        validator's warning, so the check and the behaviour cannot drift."""
        return declared_fields(self.config, self.registry)[0]

    def retriever(self) -> Any:
        r = self.config.query.retrieve
        params = dict(r.params)
        params.setdefault("on_index_error", r.on_index_error)
        reg = resolve("retrieve", r.impl, self.registry)
        return reg.build(_accepted_params(reg, params))

    def fuser(self) -> Any:
        f = self.config.query.fuse
        if not f.enabled:
            return self._build("fuse", "concat", {})
        params = dict(f.params)
        params.setdefault("k", f.k)
        params.setdefault("weights", f.weights)
        if f.weights_by_type:
            params.setdefault("weights_by_type", f.weights_by_type)
        reg = resolve("fuse", f.impl, self.registry)
        return reg.build(_accepted_params(reg, params))

    def reranker(self) -> Any:
        r = self.config.query.rerank
        if not r.enabled:
            return self._build("rerank", "noop", {})
        return self._build("rerank", r.impl, r.params)

    def path_rerankers(self) -> dict[str, Any]:
        """Rerankers a route path names for itself. The global reranker's params
        apply when the path names the same implementation."""
        r = self.config.query.rerank
        if not r.enabled:
            return {}
        out: dict[str, Any] = {}
        for name, spec in self.config.query.route.paths.items():
            if spec.rerank and spec.rerank != r.impl:
                out[name] = self._build("rerank", spec.rerank, {})
        return out

    # ------------------------------------------------------------ pipelines

    def ingestion(self, *, strict_contracts: bool = True) -> IngestionPipeline:
        c = self.config
        return IngestionPipeline(
            scanner=self.scanner(),
            parser=self.parser(),
            segmenter=self.segmenter(),
            enrichers=self.enrichers(),
            indexes=self.indexes,
            ledger=self.ledger,
            unit_store=self.unit_store,
            cache=self.cache,
            config=redact(self.resolved),
            config_hash=self.config_hash,
            strict_contracts=strict_contracts,
            enrich_enabled=c.enrich_enabled,
            segment_enabled=c.ingestion.segment.enabled,
            parse_enabled=c.ingestion.parse.enabled,
            on_document_error=c.ingestion.parse.on_error,
            index_batch_size=c.ingestion.index.batch_size,
            enrich_batch_size=c.ingestion.enrich.batch_size,
            enrich_max_concurrency=c.ingestion.enrich.max_concurrency,
            enrich_on_error=c.ingestion.enrich.on_error,
            checkpoint_every=c.ingestion.checkpoint_every,
            cache_refs=self.cache_refs,
            purge_cache=c.cache.purge_unreferenced,
            manifest_dir=self.paths.manifests,
            access_field=c.query.access.field if c.query.access.enabled else None,
        )

    def query_engine(self) -> QueryEngine:
        c = self.config
        log = (
            (c.query.route.decision_log or str(Path(self.paths.store) / "route-decisions.jsonl"))
            if c.query.route.log_decisions
            else None
        )
        return QueryEngine(
            indexes=self.indexes,
            router=self.router(),
            retriever=self.retriever(),
            fuser=self.fuser(),
            reranker=self.reranker(),
            unit_store=self.unit_store,
            cache=self.cache,
            route_enabled=c.query.route.enabled,
            fuse_enabled=c.query.fuse.enabled,
            rerank_enabled=c.query.rerank.enabled,
            rerank_input_top_k=c.query.rerank.input_top_k,
            rerank_output_top_k=c.query.rerank.output_top_k,
            default_top_k=c.query.retrieve.default_top_k,
            decision_log=log,
            path_rerankers=self.path_rerankers(),
            access=AccessPolicy(
                enabled=c.query.access.enabled,
                field=c.query.access.field,
                missing=c.query.access.missing,
            ),
            shape=ShapePolicy(
                enabled=c.query.shape.enabled,
                max_per_document=c.query.shape.max_per_document,
                collapse_duplicates=c.query.shape.collapse_duplicates,
                near_duplicate_bits=c.query.shape.near_duplicate_bits,
                expand_neighbors=c.query.shape.expand_neighbors,
                distinguish_by=tuple(c.query.shape.distinguish_by),
            ),
        )


def declared_fields(
    config: Config, registry: Registry | None = None
) -> tuple[list[str], dict[str, str]]:
    """``(names, types)`` of every field the configured stages say they write.

    The router's vocabulary. It used to be read from one enricher's params --
    the regex extractor's ``fields`` -- so what the entity extractor found, the
    facts a FatturaPA parser reads, and anything a model extracted were fields
    no question could name. Each implementation now declares what it writes
    (``declares_fields`` on its registration); the config's own declarations
    still win on type. A name without a known type is in the lexicon only: a
    wrong type is worse than none, because the router then queries a column the
    value was never written to.
    """
    types: dict[str, str] = {}
    names = set(config.extracted_field_names())
    ing = config.ingestion
    specs: list[tuple[str, str, Mapping[str, Any]]] = []
    if ing.parse.enabled:
        specs.append(("parse", ing.parse.default.impl, ing.parse.default.params))
        specs.extend(("parse", r.impl, r.params) for r in ing.parse.routes)
    if config.enrich_enabled:
        specs.extend(("enrich", e.impl, e.params) for e in ing.enrich.enrichers if e.enabled)
    for stage, impl, params in specs:
        try:
            declared = resolve(stage, impl, registry).fields(params)
        except ConfigError:
            continue  # an unknown name or bad params is phase-2 validation's report
        types.update(declared)
    types.update(config.extracted_field_types())
    names |= set(types)
    return sorted(names), types


def build_indexes(
    config: Config, store_root: str, registry: Registry | None = None
) -> dict[str, Index]:
    """Construct every enabled index.

    The fan-out that makes "a fourth kind touches nothing" true: this loop does
    not branch on ``kind``, and adding one changes nothing here.
    """
    out: dict[str, Index] = {}
    for spec in config.ingestion.index.indexes:
        if not spec.enabled:
            continue
        reg = resolve("index", spec.impl, registry)
        params = dict(spec.params)
        params.setdefault("path", str(Path(store_root) / f"{spec.name}.{spec.impl}.json"))
        idx = reg.build(_accepted_params(reg, params), name=spec.name)
        if str(idx.kind) != spec.kind:
            raise ConfigError(
                f"index {spec.name!r} is declared kind {spec.kind!r} but implementation "
                f"{spec.impl!r} reports {idx.kind!r}; routing targets would be wrong"
            )
        out[spec.name] = idx
    return out


def _accepted_params(reg: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop stage-level defaults an implementation does not accept.

    The stage config carries settings meaningful to the *stage* (max_tokens,
    fusion weights); not every implementation of that stage takes all of them.
    Passing them blindly would make an implementation that ignores one fail to
    construct. Passing an *explicitly set* unknown param is still an error --
    that is the typo case, and it is caught by ``reg.normalize``.
    """
    try:
        reg.normalize(params)
        return dict(params)
    except ConfigError:
        from dataclasses import fields as dc_fields

        model = getattr(reg.params_model, "__closure__", None)
        known: set[str] = set()
        if model:
            for cell in model:
                try:
                    if hasattr(cell.cell_contents, "__dataclass_fields__"):
                        known = {f.name for f in dc_fields(cell.cell_contents)}
                        break
                    if isinstance(cell.cell_contents, set):
                        known = set(cell.cell_contents)
                except ValueError:  # pragma: no cover - empty cell
                    continue
        filtered = {k: v for k, v in params.items() if k in known}
        reg.normalize(filtered)
        return filtered


class _MultiScanner:
    """Concatenates several sources into one corpus."""

    def __init__(self, scanners: list[Any], limit: int | None = None) -> None:
        self._scanners = scanners
        self._limit = limit

    def scan(self) -> Any:
        n = 0
        for s in self._scanners:
            for doc in s.scan():
                if self._limit is not None and n >= self._limit:
                    return
                n += 1
                yield doc

    def fingerprint(self) -> Any:
        from indexer.core.accounting import StageFingerprint
        from indexer.core.ids import hash_obj

        return StageFingerprint(
            stage="corpus",
            impl="multi",
            version="1",
            params_hash=hash_obj([s.fingerprint().key() for s in self._scanners]),
        )


def assemble(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    llm_client: Any = None,
) -> Assembly:
    cfg, resolved = load(config_path, overrides=overrides)
    return Assembly(cfg, resolved, llm_client=llm_client)


def assemble_mapping(
    raw: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    llm_client: Any = None,
) -> Assembly:
    """Assemble from a config mapping already read from disk.

    The ablation runner uses this so that every arm is derived from **one**
    snapshot of the config. Re-reading the file per arm means a config edited
    while a run is in flight produces a report whose arms came from different
    configurations -- with nothing in the output saying so. For a measurement
    tool that is the worst possible failure: the numbers still look fine.
    """
    from indexer.config.loader import interpolate, with_overrides
    from indexer.core.errors import ConfigError

    merged = with_overrides(dict(raw), overrides) if overrides else dict(raw)
    resolved = interpolate(merged, merged)
    try:
        cfg = Config.model_validate(resolved)
    except Exception as exc:
        raise ConfigError(str(exc)) from exc
    return Assembly(cfg, resolved, llm_client=llm_client)

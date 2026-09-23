"""``python -m indexer.config.check <config.yaml>`` -- validate and summarise.

Both phases by default. Phase 1 is the schema: shape, types, cross-stage
coherence. Phase 2 resolves every ``impl:`` the config names against the
registry and validates its ``params`` against what that implementation
declared.

Phase 2 used to be skipped here, on the grounds that a config should be
checkable without installing the stack it names. The cost was that a config
naming nine implementations that did not exist printed "OK" -- which is the one
answer a check must never give wrongly. Registering an implementation imports no
heavy dependency (those are imported lazily, when the implementation is built),
so phase 2 costs nothing; what is not installed is reported as a warning, by
name, rather than failing the check. ``--schema-only`` restores phase 1 alone.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indexer.config.loader import config_hash, load
from indexer.config.schema import Config
from indexer.core.errors import ConfigError

__all__ = ["Finding", "main", "resolve_all"]

#: Distribution names in ``Registration.requires`` -> the module that proves
#: they are installed, where the two differ.
_MODULE_FOR = {
    "sentence-transformers": "sentence_transformers",
    "python-docx": "docx",
    "pymupdf4llm": "pymupdf4llm",
    "extract-msg": "extract_msg",
    "PyStemmer": "Stemmer",
    "mcp": "mcp",
}


@dataclass(frozen=True, slots=True)
class Finding:
    level: str  # "error" | "warning"
    where: str
    message: str


def _specs(cfg: Config) -> Iterator[tuple[str, str, str, Mapping[str, Any]]]:
    """(where, stage, impl, params) for every implementation the config names."""
    for i, src in enumerate(cfg.corpus.sources):
        yield f"corpus.sources[{i}]", "corpus", src.impl, src.params
    ing = cfg.ingestion
    if ing.parse.enabled:
        yield "ingestion.parse.default", "parse", ing.parse.default.impl, ing.parse.default.params
        for i, r in enumerate(ing.parse.routes):
            yield f"ingestion.parse.routes[{i}]", "parse", r.impl, r.params
    if ing.segment.enabled:
        yield "ingestion.segment", "segment", ing.segment.impl, ing.segment.params
    if ing.enrich.enabled:
        for i, e in enumerate(ing.enrich.enrichers):
            if e.enabled:
                yield f"ingestion.enrich.enrichers[{i}]", "enrich", e.impl, e.params
    for spec in ing.index.indexes:
        if spec.enabled:
            yield f"ingestion.index.indexes[{spec.name}]", "index", spec.impl, spec.params
    q = cfg.query
    if q.route.enabled:
        yield "query.route", "route", q.route.impl, q.route.params
        for name, path in q.route.paths.items():
            if path.rerank:
                yield f"query.route.paths.{name}.rerank", "rerank", path.rerank, {}
    yield "query.retrieve", "retrieve", q.retrieve.impl, q.retrieve.params
    if q.fuse.enabled:
        yield "query.fuse", "fuse", q.fuse.impl, q.fuse.params
    if q.rerank.enabled:
        yield "query.rerank", "rerank", q.rerank.impl, q.rerank.params
    if cfg.eval.judge is not None:
        yield "eval.judge", "judge", cfg.eval.judge.impl, cfg.eval.judge.params
    if cfg.eval.bootstrap is not None:
        yield "eval.bootstrap", "bootstrap", cfg.eval.bootstrap.impl, cfg.eval.bootstrap.params


def resolve_all(cfg: Config) -> list[Finding]:
    """Phase 2 over a whole config: every name resolves, every params block
    validates, and every optional dependency an implementation declares is
    either installed or reported."""
    # Imported for their registrations: bootstrappers, judges, implementations.
    import indexer.eval.bootstrap
    import indexer.eval.judge
    import indexer.impls  # noqa: F401
    from indexer.core.registry import resolve

    out: list[Finding] = []
    for where, stage, impl, params in _specs(cfg):
        try:
            reg = resolve(stage, impl)
        except ConfigError as exc:
            out.append(Finding("error", where, str(exc)))
            continue
        try:
            reg.normalize(params)
        except ConfigError as exc:
            out.append(Finding("error", where, str(exc)))
        for dist in reg.requires:
            module = _MODULE_FOR.get(dist, dist.replace("-", "_"))
            if importlib.util.find_spec(module) is None:
                out.append(
                    Finding(
                        "warning",
                        where,
                        f"{stage}/{impl} needs `{dist}`, which is not installed here; "
                        f"building it will fail until it is",
                    )
                )
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    schema_only = "--schema-only" in args
    paths = [a for a in args if a != "--schema-only"]
    if not paths:
        print(
            "usage: python -m indexer.config.check [--schema-only] <config.yaml> [...]",
            file=sys.stderr,
        )
        return 2

    failed = False
    for path in paths:
        print(f"== {path}")
        try:
            cfg, resolved = load(path)
        except ConfigError as exc:
            print(f"   INVALID: {exc}\n", file=sys.stderr)
            failed = True
            continue

        idx = cfg.ingestion.index.indexes
        print(f"   project      {cfg.project.name}")
        print(f"   config_hash  {config_hash(resolved)}")
        print(
            "   ingestion    parse={} segment={} enrich={} ({} enricher(s))".format(
                cfg.ingestion.parse.default.impl,
                cfg.ingestion.segment.impl,
                "on" if cfg.enrich_enabled else "OFF",
                sum(1 for e in cfg.ingestion.enrich.enrichers if e.enabled),
            )
        )
        print(
            "   indexes      "
            + ", ".join(f"{i.name}:{i.kind}" + ("" if i.enabled else " (off)") for i in idx)
        )
        print(
            "   query        route={} retrieve={} fuse={} rerank={}".format(
                cfg.query.route.impl if cfg.query.route.enabled else "OFF",
                cfg.query.retrieve.impl,
                cfg.query.fuse.impl if cfg.query.fuse.enabled else "OFF",
                cfg.query.rerank.impl if cfg.query.rerank.enabled else "OFF",
            )
        )
        print(f"   paths        {', '.join(sorted(cfg.query.route.paths))}")
        if cfg.eval.ablations:
            print(f"   ablations    {', '.join(a.name for a in cfg.eval.ablations)}")
        declared: list[str] = []
        if not schema_only:
            from indexer.pipeline.build import declared_fields

            declared = declared_fields(cfg)[0]
        for w in cfg.warnings(declared):
            print(f"   warning      {w}")

        errors = 0
        if not schema_only:
            for f in resolve_all(cfg):
                if f.level == "error":
                    errors += 1
                    print(f"   ERROR        {f.where}: {f.message}", file=sys.stderr)
                else:
                    print(f"   warning      {f.where}: {f.message}")
        if errors:
            print(f"   INVALID ({Path(path).name}): {errors} error(s)\n", file=sys.stderr)
            failed = True
        else:
            note = " (schema only; implementations not resolved)" if schema_only else ""
            print(f"   OK ({Path(path).name}){note}\n")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

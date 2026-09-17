"""``python -m indexer.config.check <config.yaml>`` -- validate and summarise.

Phase 1 only: no implementations are imported, so a config can be checked
without installing the stack it names. That is the point -- reviewing a config
change should not require a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

from indexer.config.loader import config_hash, load
from indexer.core.errors import ConfigError


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m indexer.config.check <config.yaml> [...]", file=sys.stderr)
        return 2

    failed = False
    for path in args:
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
        for w in cfg.warnings():
            print(f"   warning      {w}")
        print(f"   OK ({Path(path).name})\n")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

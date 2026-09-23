"""``indexer`` -- build, query and serve an archive from its config.

::

    indexer check   CONFIG...             validate: schema, then every implementation
    indexer build   CONFIG [--prefill]    incremental build; --prefill answers the
                                          model calls through batches first
    indexer prefill CONFIG                the next build's model calls, by batch
    indexer query   CONFIG TEXT           one question, answered as an agent sees it
    indexer schema  CONFIG                the fields a filter can name
    indexer review  CONFIG [--out FILE]   extracted values held back for review
    indexer sweep-cache CONFIG            delete cache entries no document uses
    indexer mcp     CONFIG                serve the agent tools over MCP (stdio)

``--principal`` (repeatable) states who is asking, for configs with access
control on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import Any, TextIO

__all__ = ["main", "review_items"]


def _assemble(path: str) -> Any:
    from indexer.pipeline import assemble

    return assemble(path)


def _print(data: Any, out: TextIO) -> None:
    out.write(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def review_items(assembly: Any) -> Iterator[dict[str, Any]]:
    """Every value an enricher held back, one item per document and field.

    The review queue: an LLM extractor's value whose quote was not in the
    document, a VAT number whose check digit failed. Read from the stored units,
    so it reflects the index as built, and nothing is written at extraction
    time that a cache hit could then skip.
    """
    seen: set[tuple[str, str, str, str]] = set()
    for uid in assembly.unit_store.all_ids():
        eu = assembly.unit_store.get(uid)
        if eu is None:
            continue
        for name, e in sorted(eu.enrichments.items()):
            rejected = e.extra.get("rejected") if e.extra else None
            if not rejected:
                continue
            entries = (
                rejected
                if isinstance(rejected, list)
                else [
                    {"field": kind, "value": v, "reason": "fails its check digit"}
                    for kind, values in dict(rejected).items()
                    for v in values
                ]
            )
            for r in entries:
                key = (str(eu.document_id), name, str(r.get("field")), str(r.get("value")))
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "document_id": str(eu.document_id),
                    "source": eu.unit.metadata.get("relpath") or eu.unit.provenance.source_uri,
                    "enricher": name,
                    **{k: r.get(k) for k in ("field", "value", "evidence", "reason")},
                }


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="indexer", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="validate configs")
    p.add_argument("configs", nargs="+")
    p.add_argument("--schema-only", action="store_true")

    p = sub.add_parser("build", help="build or update the indexes")
    p.add_argument("config")
    p.add_argument("--prefill", action="store_true", help="answer model calls by batch first")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("prefill", help="the next build's model calls, through batches")
    p.add_argument("config")
    p.add_argument("--enricher", action="append", default=None)
    p.add_argument("--poll-seconds", type=float, default=60.0)

    for name, help_ in (
        ("query", "ask one question"),
        ("schema", "describe the fields"),
        ("mcp", "serve the agent tools over MCP on stdio"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config")
        if name == "query":
            p.add_argument("text")
            p.add_argument("--top-k", type=int, default=8)
        p.add_argument("--principal", action="append", default=None)

    p = sub.add_parser("review", help="list extracted values held back for review")
    p.add_argument("config")
    p.add_argument("--out", default=None)

    p = sub.add_parser("sweep-cache", help="delete cache entries no document uses")
    p.add_argument("config")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "check":
        from indexer.config.check import main as check

        return check([*args.configs, *(["--schema-only"] if args.schema_only else [])])

    a = _assemble(args.config)

    if args.command in ("build", "prefill"):
        pipe = a.ingestion()
        progress = None if getattr(args, "quiet", False) else (lambda m: print(m, file=sys.stderr))
        if args.command == "prefill" or args.prefill:
            report = pipe.prefill(
                enrichers=getattr(args, "enricher", None),
                poll_seconds=getattr(args, "poll_seconds", 60.0),
                progress=progress,
            )
            _print({"prefill": asdict(report)}, out)
            if args.command == "prefill":
                return 0
        result = pipe.build(progress=progress)
        c = result.manifest.corpus
        _print(
            {
                "build_id": result.manifest.build_id,
                "documents": {
                    "total": c.documents_total,
                    "added": c.documents_added,
                    "changed": c.documents_changed,
                    "restaged": c.documents_restaged,
                    "removed": c.documents_removed,
                    "failed": c.documents_failed,
                },
                "units": {"total": c.units_total, "written": c.units_written},
                "enrichments_failed": c.enrichments_failed,
                "cost_usd": round(result.manifest.total_cost_usd, 4),
                "failures": [str(f) for f in result.failures[:20]],
            },
            out,
        )
        return 0 if result.ok else 1

    if args.command == "sweep-cache":
        _print({"deleted": a.ingestion().sweep_cache()}, out)
        return 0

    if args.command == "review":
        items = list(review_items(a))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                for item in items:
                    fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            _print({"written": len(items), "to": args.out}, out)
        else:
            for item in items:
                out.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return 0

    from indexer.agent import AgentTools

    tools = AgentTools(a, principals=args.principal)
    if args.command == "query":
        _print(tools.search(args.text, top_k=args.top_k), out)
        return 0
    if args.command == "schema":
        _print(tools.describe_schema(), out)
        return 0

    from indexer.mcp_server import build_server

    build_server(tools).run()  # pragma: no cover - blocks on stdio
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render the ablation report's numeric sections from `ablation.json`.

Every number in `docs/ABLATION.md` that comes from a run is produced here
rather than transcribed. Three revisions of that report introduced two
transcription errors and one confounded comparison quoted from memory; a
generator removes the first failure mode entirely and makes the second visible,
because a comparison it cannot find in the artifact is one it will not print.

Usage:
    python scripts/report_table.py var/pypi/eval/ablation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def fmt(v: Any, dp: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{dp}f}"


def table(arms: list[dict[str, Any]]) -> str:
    head = (
        "| arm | P@5 | R@20 | nDCG@10 | fail@20 | correct | route | p50 |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| `{a['arm']}` | {fmt(a['p@5'])} | {fmt(a['recall@20'])} | {fmt(a['ndcg@10'])} "
        f"| {fmt(a['fail_rate'])} | {fmt(a['correct'])} | {fmt(a.get('route_accuracy'), 2)} "
        f"| {a['p50_ms']:.0f}ms |"
        for a in arms
    ]
    return "\n".join([head, *rows])


def delta(arms: dict[str, dict], base: str, treat: str, metric: str = "fail_rate") -> str:
    """One controlled comparison, refusing to print if an arm is missing."""
    if base not in arms or treat not in arms:
        return f"  {base} -> {treat}: ARM MISSING, not reported"
    b, t = arms[base][metric], arms[treat][metric]
    rel = (t - b) / b if b else float("nan")
    return f"  {base:24} -> {treat:24} {metric}: {b:.3f} -> {t:.3f}  ({rel:+.0%})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="var/pypi/eval/ablation.json")
    args = ap.parse_args()
    d = json.loads(Path(args.path).read_text())
    arms = d["arms"]
    by = {a["arm"]: a for a in arms}

    print("## Results\n")
    print(
        f"{len(arms)} arms, {d['golden']['total']} queries, {d['wall_s']:.0f}s, "
        f"${sum(a['usd/q'] * d['golden']['total'] for a in arms):.2f}.\n"
    )
    print(table(arms))

    print("\n### Controlled comparisons\n```")
    for label, base, treat in [
        ("hybrid vs its lexical half", "1-lexical-only", "3c-hybrid-svd"),
        ("embedder: hash vs LSA (solo)", "2-dense-only", "2b-svd-only"),
        ("embedder: hash vs LSA (fused)", "3-hybrid-rrf", "3c-hybrid-svd"),
        ("reranking", "5-hybrid-context", "6-hybrid-context-rerank"),
        ("reranking, no context", "3-hybrid-rrf", "3b-hybrid-rerank"),
        ("context: verbose", "3b-hybrid-rerank", "6-hybrid-context-rerank"),
        ("context: chunk-specific", "3b-hybrid-rerank", "10-context-lean"),
        ("routing", "6-hybrid-context-rerank", "7-no-router"),
        ("segmentation", "6-hybrid-context-rerank", "8-fixed-window"),
        ("fusion weights", "6-hybrid-context-rerank", "9-hybrid-weighted"),
    ]:
        print(f"{label}")
        print(delta(by, base, treat))
    print("```")

    ok = [a for a in arms if a["correct"] is not None and a["p@5"] is not None]
    if len(ok) > 2:
        print("\n### Metric correlations with end-to-end correctness\n```")
        ys = [a["correct"] for a in ok]
        for name, key in [
            ("P@5", "p@5"),
            ("recall@20", "recall@20"),
            ("nDCG@10", "ndcg@10"),
            ("fail@20", "fail_rate"),
        ]:
            print(f"  {name:12} r = {pearson([a[key] for a in ok], ys):+.3f}")
        print(f"  (n = {len(ok)} arms)")
        print("```")

    types = {}
    for a in arms:
        for qt, m in a.get("by_query_type", {}).items():
            types.setdefault(qt, {})[a["arm"]] = m
    if "structured" in types:
        print("\n### Router value, by query type\n```")
        print(f"  {'slice':12} {'n':>4}  {'router on':>10} {'router off':>11}")
        on, off = "6-hybrid-context-rerank", "7-no-router"
        for qt in ("structured", "numeric", "temporal", "factual"):
            t = types.get(qt, {})
            if on in t and off in t:
                print(
                    f"  {qt:12} {t[on]['n']:>4.0f}  {t[on]['fail_rate']:>10.3f} "
                    f"{t[off]['fail_rate']:>11.3f}"
                )
        if on in by and off in by:
            print(
                f"  {'ALL':12} {d['golden']['total']:>4}  "
                f"{by[on]['fail_rate']:>10.3f} {by[off]['fail_rate']:>11.3f}"
            )
        print("```")

    # Narrative figures the prose cites, so no number in the report is typed by
    # hand. Every one of these appeared in an earlier draft transcribed from a
    # terminal, and two of them were wrong.
    print("\n### Figures cited in the prose\n```")
    g = d["golden"]
    print(
        f"  queries                {g['total']}  "
        f"(factual {g['by_type'].get('factual', 0)}, "
        f"structured {g['by_type'].get('structured', 0)}, "
        f"temporal {g['by_type'].get('temporal', 0)}, "
        f"numeric {g['by_type'].get('numeric', 0)})"
    )
    print(f"  corpus                 {d['corpus']['documents']} docs, {d['corpus']['units']} units")
    print(f"  arms                   {len(arms)}")
    print(f"  wall                   {d['wall_s']:.0f}s")
    lat = [a["p50_ms"] for a in arms]
    p95 = [a["p95_ms"] for a in arms]
    print(
        f"  query latency          p50 {min(lat):.0f}-{max(lat):.0f}ms, "
        f"p95 {min(p95):.0f}-{max(p95):.0f}ms"
    )
    best = min(arms, key=lambda a: a["fail_rate"])
    print(f"  best arm by fail@20    {best['arm']} at {best['fail_rate']:.3f}")
    for name in ("1-lexical-only", "3c-hybrid-svd", "6-hybrid-context-rerank", "10-context-lean"):
        if name in by:
            a = by[name]
            print(
                f"  {name:22} P@5 {a['p@5']:.3f}  R@20 {a['recall@20']:.3f}  "
                f"nDCG@10 {a['ndcg@10']:.3f}  fail {a['fail_rate']:.3f}"
            )
    print("```")

    for w in d.get("contract_warnings", []):
        print(f"\n> **Contract check:** {w}")
    print()
    for v in d.get("sanity", []):
        mark = "PASS" if v["passed"] else "INVESTIGATE"
        print(
            f"- **[{mark}]** {v['name']}: {v['metric']} {v['baseline']:.3f} -> "
            f"{v['treatment']:.3f} ({v['observed_reduction']:+.0%}, expected "
            f"{-v['expected_reduction']:.0%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

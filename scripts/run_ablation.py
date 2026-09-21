#!/usr/bin/env python3
"""Build the index, bootstrap a golden set, run the ablation, write the report.

One command, because invariant 6 only holds if measuring is cheap enough to
actually do. Each step is skipped when its output already exists, so iterating
on the ablation does not re-pay for the corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import indexer.eval.bootstrap  # noqa: F401  -- imported for its registrations
from indexer.config.loader import load
from indexer.core.registry import resolve
from indexer.eval.golden import load_golden_set, write_golden_set
from indexer.eval.judge import ContainmentJudge
from indexer.eval.metrics import Matcher
from indexer.eval.runner import AblationRunner, EvalRunner
from indexer.pipeline.build import assemble


def say(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--golden", default=None, help="override eval.golden_set")
    ap.add_argument("--regenerate-golden", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--arms", nargs="*", default=None, help="run only these arms")
    args = ap.parse_args()

    cfg, _ = load(args.config)
    golden_path = Path(args.golden or cfg.eval.golden_set)

    # ---- 1. build the base index ---------------------------------------
    t0 = time.perf_counter()
    say("== building base index")
    base = assemble(args.config)
    res = base.ingestion().build()
    say(f"   {res.summary()}")
    say(f"   reading-order confidence: {res.manifest.corpus.mean_reading_order_confidence:.3f}")
    for name, st in sorted(res.manifest.stage_totals.items()):
        say(
            f"   {name:26} runs={st['runs']:.0f} wall={st['wall_ms'] / 1000:6.1f}s "
            f"cache_hits={st['cache_hits']:.0f} errors={st['errors']:.0f}"
        )

    # ---- 1b. contract checks on the built corpus ------------------------
    # Run before evaluating, because a golden set scored against a corpus whose
    # provenance is broken produces numbers that look fine and mean nothing.
    from indexer.eval.checks import check_context_specificity

    built_units = [
        eu for uid in base.unit_store.all_ids() if (eu := base.unit_store.get(uid)) is not None
    ]
    ctx_problems = check_context_specificity(built_units)
    say("== contract checks")
    if ctx_problems:
        for c in ctx_problems:
            say(f"   WARN  {c}")
    else:
        say("   context specificity: OK")

    # ---- 2. golden set --------------------------------------------------
    if golden_path.exists() and not args.regenerate_golden:
        golden = load_golden_set(golden_path)
        say(f"== golden set: {len(golden)} queries (existing) {golden.stats()}")
    else:
        say("== bootstrapping golden set")
        import json as _json

        from indexer.core.cache import cache_key
        from indexer.pipeline.codec import decode_parsed_document

        parser_fp = base.parser().fingerprint()
        docs = []
        for sdoc in base.scanner().scan():
            raw = base.cache.get(cache_key(parser_fp, sdoc.content_hash))
            if raw:
                docs.append(decode_parsed_document(_json.loads(raw)))
        units = [
            eu for uid in base.unit_store.all_ids() if (eu := base.unit_store.get(uid)) is not None
        ]
        say(f"   {len(docs)} parsed docs, {len(units)} units")

        # Resolved by name like every other stage. Hardcoding the heuristic
        # generator here made `eval.bootstrap.impl` a config key that silently
        # did nothing -- including in configs/full.yaml, which names
        # `llm_bootstrap` and was getting the heuristic one.
        spec = cfg.eval.bootstrap
        reg = resolve("bootstrap", spec.impl if spec else "heuristic")
        say(f"   generator: {reg.stage}/{reg.name}")
        boot = reg.build(spec.params if spec else {})
        golden = boot.bootstrap(docs, units, baseline=base.indexes.get("lexical"))
        write_golden_set(golden, golden_path)
        say(f"   wrote {len(golden)} queries -> {golden_path}")
        say(f"   {golden.stats()}")
        say(f"   {golden.notes}")

    if not golden.queries:
        say("!! empty golden set; nothing to evaluate")
        return 1

    # ---- 3. ablation ----------------------------------------------------
    arms = list(cfg.eval.ablations)
    if args.arms:
        arms = [a for a in arms if a.name in args.arms]
    say(f"== ablation: {len(arms)} arms")

    runner = EvalRunner(
        matcher=Matcher(policy=cfg.eval.match, min_overlap=cfg.eval.min_overlap),
        k_values=cfg.eval.k_values,
        failure_k=cfg.eval.failure_k,
        top_k=max([*cfg.eval.k_values, cfg.eval.failure_k]),
        judge=ContainmentJudge({}),
    )
    ab = AblationRunner(
        config_path=args.config,
        golden=golden,
        runner=runner,
        baseline_arm=arms[0].name if arms else "baseline",
        progress=say,
    )
    result = ab.run(arms, cfg.eval.sanity_checks)

    say("")
    say(result.delta_table())

    # ---- 4. report ------------------------------------------------------
    out_dir = Path(args.report or cfg.eval.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "corpus": {
                    "documents": res.manifest.corpus.documents_total,
                    "units": res.manifest.corpus.units_total,
                    "bytes": res.manifest.corpus.bytes_parsed,
                },
                "golden": golden.stats(),
                "contract_warnings": ctx_problems,
                "wall_s": time.perf_counter() - t0,
                "arms": [
                    {
                        **r.headline(),
                        "config_hash": r.config_hash,
                        "recall": r.recall,
                        "precision": r.precision,
                        "ndcg": r.ndcg,
                        "mrr": r.mrr,
                        "route_accuracy": r.route_accuracy,
                        "by_query_type": r.by_query_type,
                        "errors": r.errors,
                        "notes": r.notes,
                    }
                    for r in result.reports
                ],
                "sanity": [
                    {
                        "name": v.name,
                        "metric": v.metric,
                        "baseline": v.baseline_value,
                        "treatment": v.treatment_value,
                        "observed_reduction": v.observed_reduction,
                        "expected_reduction": v.expected_reduction,
                        "passed": v.passed,
                        "diagnosis": v.diagnosis,
                    }
                    for v in result.verdicts
                ],
            },
            indent=2,
            default=str,
        )
    )
    (out_dir / "ablation.txt").write_text(result.delta_table())
    say(f"\nwrote {out_dir}/ablation.json and ablation.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

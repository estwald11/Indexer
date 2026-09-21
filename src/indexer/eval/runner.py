"""The evaluation runner and the ablation runner.

Two jobs. The runner scores one system against a golden set. The ablation runner
builds and scores several arms and prints the delta table.

The costing decision that makes ablations affordable: an arm that overrides only
query-side keys reuses the built index. Rebuilding for every arm is how invariant
6 stops being followed in practice -- if measuring costs an hour, measurement
stops happening.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indexer.config.schema import AblationSpec, SanityCheck
from indexer.core.query import Query, QueryType, RoutePath
from indexer.core.results import Hit, RetrievalResponse
from indexer.eval.golden import GoldenQuery, GoldenSet
from indexer.eval.harness import AblationResult, SanityVerdict
from indexer.eval.metrics import (
    Matcher,
    QueryScore,
    RunReport,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_failed,
)

__all__ = ["AblationRunner", "EvalRunner", "default_answer"]


def default_answer(response: RetrievalResponse) -> str:
    """Concatenate the top passages.

    The frame does not generate. End-to-end correctness still needs *something*
    to judge, so the default is the trivial answerer and a caller with a real
    generator passes one in. Scores from this default measure whether the
    evidence was retrieved, not whether a model used it well -- which is the
    right thing for a retrieval library to report, and invariant 1 says it is
    the part that matters anyway.
    """
    if response.records is not None:
        return json.dumps(response.records.as_dicts()[:20], default=str)
    return "\n\n".join((h.unit.unit.text if h.unit else h.matched_text) for h in response.hits[:5])


@dataclass(slots=True)
class EvalRunner:
    """Scores a query engine against a golden set."""

    matcher: Matcher = field(default_factory=Matcher)
    k_values: Sequence[int] = (1, 5, 10, 20)
    failure_k: int = 20
    top_k: int = 20
    answerer: Callable[[RetrievalResponse], str] = default_answer
    judge: Any = None

    def run(self, engine: Any, golden: GoldenSet, *, arm: str = "baseline") -> RunReport:
        scores: list[QueryScore] = []
        for item in golden:
            scores.append(self._score_one(engine, item))
        return self._aggregate(arm, scores, golden)

    # ------------------------------------------------------------------ one

    def _score_one(self, engine: Any, item: GoldenQuery) -> QueryScore:
        t0 = time.perf_counter()
        try:
            resp = engine.execute(Query(text=item.query, top_k=self.top_k))
        except Exception as exc:
            return QueryScore(query_id=item.id, failed=True, error=f"{type(exc).__name__}: {exc}")
        elapsed = (time.perf_counter() - t0) * 1000

        hits: Sequence[Hit] = resp.hits
        grades = self.matcher.grades(hits, item)
        ideal = sorted((r.weight for r in item.relevant), reverse=True)
        n_rel = len(item.relevant)

        s = QueryScore(
            query_id=item.id,
            recall={k: recall_at_k(grades, n_rel, k) for k in self.k_values},
            precision={k: precision_at_k(grades, k) for k in self.k_values},
            ndcg={k: ndcg_at_k(grades, ideal, k) for k in self.k_values},
            mrr=reciprocal_rank(grades),
            failed=retrieval_failed(grades, self.failure_k),
            routed_path=str(resp.decision.path),
            latency_ms=elapsed,
            cost_usd=resp.cost_usd,
            hits_returned=len(hits),
        )
        s.route_correct = _route_correct(item.query_type, resp)

        # The structured path returns records, not passages. Span-overlap
        # metrics would report 0 for a correct answer, so they are marked
        # not-applicable rather than counted as failures.
        #
        # But an *empty* record set is a failure, and saying otherwise was a bug
        # that flattered every arm with no extracted fields: those arms routed
        # 33 structured questions to an index containing nothing, got nothing
        # back, and were scored as having answered all 33. The failure rate that
        # produced was 0.075 where the honest figure is 0.141. A metric that
        # rewards a stage for being asked rather than for answering will make
        # any ablation involving it meaningless.
        if resp.records is not None:
            s.failed = not resp.records.rows
            s.recall = {k: float("nan") for k in self.k_values}
            s.precision = {k: float("nan") for k in self.k_values}
            s.ndcg = {k: float("nan") for k in self.k_values}

        if self.judge is not None:
            s.correct = self.judge.judge(item, self.answerer(resp), hits)
        return s

    # ------------------------------------------------------------ aggregate

    def _aggregate(self, arm: str, scores: list[QueryScore], golden: GoldenSet) -> RunReport:
        ok = [s for s in scores if s.error is None]
        rep = RunReport(arm=arm, n_queries=len(scores), scores=scores)
        rep.errors = sum(1 for s in scores if s.error)
        if not ok:
            return rep

        rep.recall = {k: _mean_defined(s.recall.get(k) for s in ok) for k in self.k_values}
        rep.precision = {k: _mean_defined(s.precision.get(k) for s in ok) for k in self.k_values}
        rep.ndcg = {k: _mean_defined(s.ndcg.get(k) for s in ok) for k in self.k_values}
        rep.mrr = _mean_defined(s.mrr for s in ok)
        rep.retrieval_failure_rate = sum(1 for s in ok if s.failed) / len(ok)

        routed = [s for s in ok if s.route_correct is not None]
        rep.route_accuracy = (
            sum(1 for s in routed if s.route_correct) / len(routed) if routed else None
        )
        judged = [s for s in ok if s.correct is not None]
        rep.correctness = sum(1 for s in judged if s.correct) / len(judged) if judged else None

        lat = sorted(s.latency_ms for s in ok)
        rep.latency_p50_ms = _pct(lat, 50)
        rep.latency_p95_ms = _pct(lat, 95)
        rep.cost_per_query_usd = sum(s.cost_usd for s in ok) / len(ok)

        # The per-type slice. Aggregates hide the invariant-5 failure mode:
        # structured questions are a minority, so misrouting them all costs a
        # couple of points overall and is invisible, while being catastrophic
        # for that slice.
        by_id = {q.id: q for q in golden}
        by_type: dict[str, list[QueryScore]] = {}
        for s in ok:
            item = by_id.get(s.query_id)
            if item:
                by_type.setdefault(str(item.query_type), []).append(s)
        rep.by_query_type = {
            t: {
                "n": float(len(group)),
                "p@5": _mean_defined(g.precision.get(5) for g in group),
                "recall@20": _mean_defined(g.recall.get(20) for g in group),
                "fail_rate": sum(1 for g in group if g.failed) / len(group),
                "route_acc": _mean_defined(
                    (1.0 if g.route_correct else 0.0) for g in group if g.route_correct is not None
                ),
            }
            for t, group in sorted(by_type.items())
        }
        if not golden.verified_only().queries:
            rep.notes.append(
                "golden set is entirely UNVERIFIED (machine-generated); treat absolute "
                "values as indicative and compare arms only"
            )
        return rep


def _route_correct(expected: QueryType, resp: RetrievalResponse) -> bool | None:
    """Did the router send this query down the right path?

    Only scored where the golden item states a type that implies a path.
    Factual questions can legitimately take LOOKUP or ITERATIVE, so they are
    not scored rather than scored arbitrarily.
    """
    path = str(resp.decision.path)
    if expected in (QueryType.STRUCTURED, QueryType.NUMERIC, QueryType.TEMPORAL):
        return path == RoutePath.STRUCTURED
    if expected in (QueryType.MULTI_HOP, QueryType.COMPARATIVE):
        return path == RoutePath.ITERATIVE
    return None


def _mean_defined(values: Any) -> float:
    vals = [v for v in values if v is not None and v == v]  # drop None and NaN
    return sum(vals) / len(vals) if vals else float("nan")


def _pct(sorted_values: list[float], p: int) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    return float(statistics.quantiles(sorted_values, n=100)[min(p, 99) - 1])


# --------------------------------------------------------------------------- #
# ablation                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AblationRunner:
    """Builds and evaluates each arm, then prints the delta table."""

    config_path: str | Path
    golden: GoldenSet
    runner: EvalRunner = field(default_factory=EvalRunner)
    baseline_arm: str = "baseline"
    progress: Callable[[str], None] | None = None
    #: Arms that only change query-side keys reuse the built index. Without
    #: this, measuring costs a full rebuild per arm and stops happening.
    reuse_index: bool = True

    def run(
        self,
        arms: Sequence[AblationSpec],
        sanity_checks: Sequence[SanityCheck] = (),
    ) -> AblationResult:
        from indexer.config.loader import load_mapping
        from indexer.pipeline.build import assemble_mapping

        # One snapshot for the whole run. Every arm is derived from it, so the
        # arms provably differ only in their stated overrides even if the file
        # changes underneath a long run.
        snapshot = load_mapping(self.config_path)

        t0 = time.perf_counter()
        result = AblationResult(baseline_arm=self.baseline_arm)
        built: dict[str, str] = {}  # ingestion fingerprint -> arm that built it

        for spec in arms:
            self._say(f"--- arm: {spec.name}")
            overrides = dict(spec.overrides)

            # Each distinct ingestion configuration gets its own store. Sharing
            # one store across arms makes reuse depend on the order the arms
            # happen to run in: an arm that "reuses the index" would silently
            # read whatever the previous arm left behind. Keying the store by
            # the ingestion configuration makes reuse a fact about the config
            # rather than about the loop.
            probe = assemble_mapping(snapshot, overrides=overrides)
            ing_fp = _ingestion_fingerprint(probe.resolved)
            if self.reuse_index:
                overrides = {
                    **overrides,
                    "paths.store": str(Path(probe.paths.store) / "arms" / ing_fp[:16]),
                }
            assembly = assemble_mapping(snapshot, overrides=overrides)

            prior = built.get(ing_fp)
            if prior is not None and self.reuse_index:
                self._say(f"    build: reused (same ingestion as {prior!r})")
            else:
                res = assembly.ingestion().build()
                self._say(f"    build: {res.summary()}")
                result.total_cost_usd += res.manifest.total_cost_usd
                built[ing_fp] = spec.name

            report = self.runner.run(assembly.query_engine(), self.golden, arm=spec.name)
            report.config_hash = assembly.config_hash
            if spec.description:
                report.notes.append(spec.description)
            result.reports.append(report)
            result.total_cost_usd += report.cost_per_query_usd * report.n_queries
            self._say(
                f"    P@5={report.precision.get(5, float('nan')):.3f} "
                f"fail={report.retrieval_failure_rate:.3f} "
                f"route_acc={report.route_accuracy}"
            )

        result.verdicts = [self._verdict(c, result) for c in sanity_checks]
        result.total_wall_s = time.perf_counter() - t0
        return result

    def _verdict(self, check: SanityCheck, result: AblationResult) -> SanityVerdict:
        by_arm = {r.arm: r for r in result.reports}
        base, treat = by_arm.get(check.baseline_arm), by_arm.get(check.treatment_arm)
        if base is None or treat is None:
            return SanityVerdict(
                name=check.name,
                baseline_arm=check.baseline_arm,
                treatment_arm=check.treatment_arm,
                metric=check.metric,
                baseline_value=float("nan"),
                treatment_value=float("nan"),
                observed_reduction=float("nan"),
                expected_reduction=check.expected_reduction,
                tolerance=check.tolerance,
                passed=False,
                diagnosis=f"arm missing: {check.baseline_arm!r} or {check.treatment_arm!r}",
            )
        bv = _metric(base, check.metric)
        tv = _metric(treat, check.metric)
        observed = (tv - bv) / bv if bv else float("nan")
        lo = -check.expected_reduction * (1 + check.tolerance)
        hi = -check.expected_reduction * (1 - check.tolerance)
        passed = lo <= observed <= hi
        return SanityVerdict(
            name=check.name,
            baseline_arm=check.baseline_arm,
            treatment_arm=check.treatment_arm,
            metric=check.metric,
            baseline_value=bv,
            treatment_value=tv,
            observed_reduction=observed,
            expected_reduction=check.expected_reduction,
            tolerance=check.tolerance,
            passed=passed,
            diagnosis=_diagnose(observed, check, bv),
        )

    def _say(self, msg: str) -> None:
        if self.progress:
            self.progress(msg)


#: Config sections that determine what ends up in an index. Two arms agreeing
#: on all of these can share one build; disagreeing anywhere means a rebuild.
_INGESTION_SECTIONS = ("corpus", "ingestion")


def _ingestion_fingerprint(resolved: Mapping[str, Any]) -> str:
    from indexer.core.ids import hash_obj, short

    return short(hash_obj({k: resolved.get(k) for k in _INGESTION_SECTIONS}), 32)


def _metric(rep: RunReport, name: str) -> float:
    if name == "retrieval_failure_rate":
        return rep.retrieval_failure_rate
    if name.startswith("precision@"):
        return rep.precision.get(int(name.split("@")[1]), float("nan"))
    if name.startswith("recall@"):
        return rep.recall.get(int(name.split("@")[1]), float("nan"))
    if name.startswith("ndcg@"):
        return rep.ndcg.get(int(name.split("@")[1]), float("nan"))
    if name == "correctness":
        return rep.correctness if rep.correctness is not None else float("nan")
    raise ValueError(f"unknown metric {name!r}")


def _diagnose(observed: float, check: SanityCheck, baseline_value: float) -> str:
    """The checklist from SanityVerdict, narrowed by what was actually seen."""
    if observed != observed:
        return "metric unavailable in one arm"
    if baseline_value < 0.02:
        return (
            f"baseline {check.metric} is already {baseline_value:.3f} -- there is almost no "
            f"headroom, so this delta is noise. The golden set is too easy: raise "
            f"eval.bootstrap.drop_if_baseline_rank or use harder queries."
        )
    if observed > 0:
        return (
            "the treatment made things WORSE. Check that the change is wired at all "
            "(compare the two arms' config_hash), then that the reranker's input_top_k "
            "is at least as deep as the failures it is meant to fix."
        )
    if abs(observed) < check.expected_reduction * (1 - check.tolerance):
        return (
            "smaller than expected. Most likely: one index is indexing unit.text rather "
            "than indexing_text(), so only half the hybrid sees the context -- run "
            "indexer.eval.checks.check_index_surface against each index. Also check that "
            "gold spans still line up with the corpus (RelevantSpan.snippet)."
        )
    return "larger than expected -- check the golden set is not leaking the treatment's own signal"

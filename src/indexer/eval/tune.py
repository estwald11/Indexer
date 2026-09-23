"""Fusion weights, fitted on a dev split and reported on a test split.

RRF's per-index weights are the query path's one knob that is a number to
choose rather than a component to swap, and the default -- every index at 1.0
-- is a prior, not a measurement. ``tune_fusion`` searches a small grid on the
dev half of a golden set, overall and per query type (``weights_by_type``), and
scores the choice on the test half, which had no say in it. Both numbers are
reported: a gain on dev that does not survive on test is the tuning fitting the
queries it was shown.

Per query type means the type the *router predicts*, because that is what the
fuser is told at query time. Grouping by the golden label would tune weights
for a type the fuser never sees when the router is wrong.

Query-time only: nothing is rebuilt, so a grid of twenty-five candidates costs
twenty-five passes over the dev queries.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from indexer.eval.golden import GoldenSet
from indexer.eval.metrics import QueryScore, RunReport

__all__ = ["TuneResult", "tune_fusion"]

Weights = dict[str, float]


@dataclass(slots=True)
class TuneResult:
    metric: str
    weights: Weights
    higher_is_better: bool = True
    weights_by_type: dict[str, Weights] = field(default_factory=dict)
    dev_default: float = math.nan
    dev_tuned: float = math.nan
    test_default: float = math.nan
    test_tuned: float = math.nan
    #: Every candidate tried on dev, with its overall score.
    trials: list[tuple[Weights, float]] = field(default_factory=list)

    @property
    def holds_on_test(self) -> bool:
        gain = self.test_tuned - self.test_default
        return gain > 0 if self.higher_is_better else gain < 0

    def render(self) -> str:
        lines = [
            f"fusion weights tuned on dev for {self.metric}",
            f"  dev   default {self.dev_default:.4f}  tuned {self.dev_tuned:.4f}",
            f"  test  default {self.test_default:.4f}  tuned {self.test_tuned:.4f}"
            + ("" if self.holds_on_test else "   (does not hold on test: keep the default)"),
            "",
            "query:",
            "  fuse:",
            "    weights: {" + ", ".join(f"{k}: {v}" for k, v in self.weights.items()) + "}",
        ]
        if self.weights_by_type:
            lines.append("    weights_by_type:")
            for t, w in sorted(self.weights_by_type.items()):
                lines.append(f"      {t}: {{" + ", ".join(f"{k}: {v}" for k, v in w.items()) + "}")
        return "\n".join(lines)


def _reader(metric: str) -> tuple[Callable[[QueryScore], float], bool]:
    """How to read ``metric`` from one query's score, and whether higher is better."""
    name, _, k = metric.partition("@")
    at = int(k) if k else 10
    readers: dict[str, tuple[Callable[[QueryScore], float], bool]] = {
        "ndcg": (lambda s: s.ndcg.get(at, math.nan), True),
        "recall": (lambda s: s.recall.get(at, math.nan), True),
        "precision": (lambda s: s.precision.get(at, math.nan), True),
        "mrr": (lambda s: s.mrr, True),
        "fail": (lambda s: 1.0 if s.failed else 0.0, False),
    }
    if name not in readers:
        raise ValueError(f"metric must be one of {sorted(readers)} (with @k), not {metric!r}")
    return readers[name]


def _mean(scores: Sequence[QueryScore], read: Callable[[QueryScore], float]) -> float:
    values = [v for s in scores if s.error is None and not math.isnan(v := read(s))]
    return sum(values) / len(values) if values else math.nan


def _distance(weights: Mapping[str, float]) -> float:
    """How far weights are from all-equal, for breaking ties toward the default."""
    return sum(abs(math.log(w)) for w in weights.values())


def tune_fusion(
    engine: Any,
    runner: Any,
    dev: GoldenSet,
    test: GoldenSet,
    *,
    indexes: Sequence[str],
    grid: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    metric: str = "ndcg@10",
    k: int = 60,
    by_type: bool = True,
    min_type_queries: int = 8,
    min_gain: float = 0.005,
    fuser_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> TuneResult:
    """Search RRF weights on ``dev``; report default and tuned on ``test``.

    The first index keeps weight 1.0 -- RRF ranks by ratios, so fixing one
    removes a direction the grid would otherwise search for nothing. A
    per-type weight is kept only where it beats the overall choice on that
    type's dev queries by ``min_gain`` and the type has ``min_type_queries``
    of them: a type seen twice has no weights worth fitting.

    ``fuser_factory(params)`` builds a fuser; by default the engine's own
    fuser class, so this module needs no implementation imported.
    """
    read, higher = _reader(metric)
    sign = 1.0 if higher else -1.0
    make = fuser_factory or (lambda params: type(engine.fuser)(params))
    original = engine.fuser

    def run(golden: GoldenSet, weights: Weights, by: Mapping[str, Weights]) -> RunReport:
        engine.fuser = make({"k": k, "weights": dict(weights), "weights_by_type": dict(by)})
        try:
            report: RunReport = runner.run(engine, golden, arm="tune")
            return report
        finally:
            engine.fuser = original

    default = dict.fromkeys(indexes, 1.0)
    rest = list(indexes[1:])
    candidates = [
        {indexes[0]: 1.0, **dict(zip(rest, combo, strict=True))}
        for combo in itertools.product(grid, repeat=len(rest))
    ]
    if default not in candidates:
        candidates.insert(0, default)

    trials: list[tuple[Weights, float, RunReport]] = []
    for weights in candidates:
        report = run(dev, weights, {})
        trials.append((weights, _mean(report.scores, read), report))

    def score(t: tuple[Weights, float, RunReport]) -> tuple[float, float]:
        value = t[1] if not math.isnan(t[1]) else -math.inf * sign
        # Ties go to the candidate nearest the default: a weight that buys
        # nothing on dev should not be carried into production.
        return (sign * value, -_distance(t[0]))

    best_weights, _, best_report = max(trials, key=score)
    dev_default = next(v for w, v, _ in trials if w == default)

    chosen_by_type: dict[str, Weights] = {}
    if by_type:
        types = sorted({s.predicted_type for s in best_report.scores if s.predicted_type})
        for qtype in types:
            ids = {s.query_id for s in best_report.scores if s.predicted_type == qtype}
            if len(ids) < min_type_queries:
                continue
            per = [
                (w, _mean([s for s in r.scores if s.query_id in ids], read)) for w, _, r in trials
            ]
            per = [(w, v) for w, v in per if not math.isnan(v)]
            if not per:
                continue
            w_type, v_type = max(per, key=lambda p: (sign * p[1], -_distance(p[0])))
            v_best = next(v for w, v in per if w == best_weights)
            if sign * (v_type - v_best) >= min_gain:
                chosen_by_type[qtype] = w_type

    dev_tuned = _mean(run(dev, best_weights, chosen_by_type).scores, read)
    test_default = _mean(run(test, default, {}).scores, read)
    test_tuned = _mean(run(test, best_weights, chosen_by_type).scores, read)
    return TuneResult(
        metric=metric,
        weights=best_weights,
        higher_is_better=higher,
        weights_by_type=chosen_by_type,
        dev_default=dev_default,
        dev_tuned=dev_tuned,
        test_default=test_default,
        test_tuned=test_tuned,
        trials=[(w, v) for w, v, _ in trials],
    )

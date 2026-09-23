"""Whether a difference between two arms is a difference.

A delta table says arm B's failure rate is 3 points lower than arm A's. On a
golden set of 150 queries that is four or five queries, and whether it means
anything depends on which ones: the same four queries flipping every time is a
real effect, four different ones flipping at random is noise. Both tables look
identical. This module is what tells them apart, with the two tests that fit
paired evaluations on one query set:

*Paired bootstrap* for graded metrics (nDCG, recall, MRR). The queries are
resampled with replacement, the *difference* is recomputed on each resample,
and the interval is read off the percentiles. Paired because both arms answered
the same queries: the per-query difference cancels how hard each query is,
which is most of the variance an unpaired interval would carry.

*McNemar's test* for pass/fail outcomes (retrieval failure, correctness). Only
the discordant queries carry information -- failed in A and passed in B, or the
reverse -- and the test asks whether those split more unevenly than a coin
would. Exact (binomial) below 25 discordant pairs, where the chi-square
approximation is poor, which is exactly the size of a typical delta.

Both need per-query scores paired by query id; queries missing from either arm
are dropped from the comparison and counted, never imputed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from indexer.eval.metrics import QueryScore, RunReport

__all__ = ["Comparison", "compare", "mcnemar", "paired_bootstrap"]


@dataclass(frozen=True, slots=True)
class Comparison:
    """One metric, one arm against the baseline."""

    metric: str
    n: int
    #: Treatment minus baseline, on the paired queries.
    delta: float
    #: Bootstrap interval for graded metrics; None for pass/fail ones.
    low: float | None = None
    high: float | None = None
    #: McNemar's discordant counts: baseline-only passes, treatment-only passes.
    only_base: int | None = None
    only_treatment: int | None = None
    p_value: float | None = None

    @property
    def significant(self) -> bool:
        """At the 5% level: the interval excludes zero, or p < 0.05."""
        if self.low is not None and self.high is not None:
            return self.low > 0 or self.high < 0
        return self.p_value is not None and self.p_value < 0.05

    def render(self) -> str:
        star = "*" if self.significant else " "
        if self.low is not None and self.high is not None:
            return f"{self.metric:<10} {self.delta:+.3f} [{self.low:+.3f}, {self.high:+.3f}]{star}"
        return (
            f"{self.metric:<10} {self.delta:+.3f} p={self.p_value:.3f}{star} "
            f"(only base {self.only_base}, only this {self.only_treatment})"
        )


def paired_bootstrap(
    base: Sequence[float],
    treatment: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(mean difference, low, high)`` of ``treatment - base``, paired by
    position, with a percentile interval. Deterministic for a given seed."""
    if len(base) != len(treatment):
        raise ValueError("paired samples must have the same length")
    diffs = [t - b for b, t in zip(base, treatment, strict=True)]
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    lo = means[math.floor(tail * (resamples - 1))]
    hi = means[math.ceil((1 - tail) * (resamples - 1))]
    return (mean, lo, hi)


def mcnemar(base: Sequence[bool], treatment: Sequence[bool]) -> tuple[int, int, float]:
    """``(only_base, only_treatment, two-sided p)`` for paired pass/fail outcomes.

    ``True`` is a pass. Exact binomial when the discordant pairs are fewer than
    25, chi-square with continuity correction otherwise.
    """
    if len(base) != len(treatment):
        raise ValueError("paired samples must have the same length")
    b = sum(1 for x, y in zip(base, treatment, strict=True) if x and not y)
    c = sum(1 for x, y in zip(base, treatment, strict=True) if y and not x)
    n = b + c
    if n == 0:
        return (b, c, 1.0)
    if n < 25:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
        return (b, c, min(1.0, 2 * tail))
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Survival function of chi-square with one degree of freedom.
    return (b, c, math.erfc(math.sqrt(chi2 / 2)))


#: (metric name, how to read it from a score, graded?) -- pass/fail metrics
#: read as "passed" so that McNemar's only_treatment counts improvements.
_METRICS: tuple[tuple[str, Callable[[QueryScore], float | bool | None], bool], ...] = (
    ("nDCG@10", lambda s: s.ndcg.get(10), True),
    ("R@20", lambda s: s.recall.get(20), True),
    ("MRR", lambda s: s.mrr, True),
    ("found@20", lambda s: not s.failed, False),
    ("correct", lambda s: s.correct, False),
)


def compare(
    base: RunReport, treatment: RunReport, *, resamples: int = 2000, seed: int = 0
) -> list[Comparison]:
    """Every metric both arms scored, paired by query id.

    A query is left out of a metric where either arm has no value for it --
    NaN retrieval metrics on the structured path, ``None`` correctness where
    the judge abstained -- so each comparison is over the queries it can
    speak for.
    """
    by_id = {s.query_id: s for s in base.scores if s.error is None}
    pairs = [
        (by_id[s.query_id], s) for s in treatment.scores if s.query_id in by_id and s.error is None
    ]
    out: list[Comparison] = []
    for name, read, graded in _METRICS:
        values = [(read(b), read(t)) for b, t in pairs]
        kept = [
            (x, y)
            for x, y in values
            if x is not None
            and y is not None
            and not (isinstance(x, float) and math.isnan(x))
            and not (isinstance(y, float) and math.isnan(y))
        ]
        if not kept:
            continue
        if graded:
            delta, lo, hi = paired_bootstrap(
                [float(x) for x, _ in kept],
                [float(y) for _, y in kept],
                resamples=resamples,
                seed=seed,
            )
            out.append(Comparison(name, len(kept), delta, low=lo, high=hi))
        else:
            xs = [bool(x) for x, _ in kept]
            ys = [bool(y) for _, y in kept]
            only_b, only_t, p = mcnemar(xs, ys)
            delta = (sum(ys) - sum(xs)) / len(kept)
            out.append(
                Comparison(
                    name, len(kept), delta, only_base=only_b, only_treatment=only_t, p_value=p
                )
            )
    return out

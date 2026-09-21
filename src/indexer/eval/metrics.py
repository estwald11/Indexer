"""Metric definitions and the span-matching rule underneath all of them.

The metric set is fixed by the brief: recall@k, precision@5, nDCG, retrieval
failure rate, end-to-end correctness, p50/p95 latency, cost per query. What is
*not* fixed, and matters more, is the relevance judgement they all sit on.

Matching
--------
A hit is judged against gold by comparing spans, under one of three policies:

``span_overlap`` (default)
    Relevant if ``|hit ∩ gold| / |gold| >= min_overlap``. Tolerant of a chunker
    that splits a gold passage across two units -- each gets partial credit for
    the fraction it covers, and neither is punished for the boundary.

``span_containment``
    Relevant only if the hit contains the gold span entirely. Strict, and the
    right choice when a partial passage would produce a wrong answer -- a
    truncated table row, half a clause.

``unit_id``
    Exact unit id. Only valid when gold was generated against the same
    segmentation. Included because it is the cheapest check when comparing two
    arms that differ only downstream of segment, and it is exact.

Why retrieval failure rate is separate from recall
--------------------------------------------------
Recall@20 averages coverage. Retrieval failure rate counts the queries with
**nothing** relevant in the top k -- the ones that cannot be answered no matter
how good generation is. Invariant 1 is about those: retrieval failures drive
11-46% of end-to-end errors while utilisation failures stay at 4-8%. A change
that lifts mean recall from 0.71 to 0.74 while leaving the failure rate flat has
improved nothing that matters, and only reporting both makes that visible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from indexer.core.provenance import Span
from indexer.core.results import Hit
from indexer.eval.golden import GoldenQuery, RelevantSpan

__all__ = [
    "MatchPolicy",
    "Matcher",
    "QueryScore",
    "RunReport",
    "dcg",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]

MatchPolicy = str  # "span_overlap" | "span_containment" | "unit_id"


@dataclass(frozen=True, slots=True)
class Matcher:
    """Decides whether a hit satisfies a gold span."""

    policy: MatchPolicy = "span_overlap"
    min_overlap: float = 0.5

    def grade(self, hit: Hit, gold: RelevantSpan) -> int:
        """Relevance grade of this hit for this gold span; 0 if unmatched."""
        if hit.document_id != gold.document_id:
            return 0
        hspan: Span = hit.provenance.span
        match self.policy:
            case "unit_id":
                return gold.weight if hspan == gold.span else 0
            case "span_containment":
                return gold.weight if hspan.contains(gold.span) else 0
            case "span_overlap":
                if gold.span.length == 0:
                    return gold.weight if hspan.contains(gold.span) else 0
                covered = hspan.overlap_length(gold.span) / gold.span.length
                return gold.weight if covered >= self.min_overlap else 0
            case _:
                raise ValueError(f"unknown match policy {self.policy!r}")

    def grades(self, hits: Sequence[Hit], item: GoldenQuery) -> list[int]:
        """Per-hit relevance grade, best over the query's gold spans."""
        return [max((self.grade(h, g) for g in item.relevant), default=0) for h in hits]


def recall_at_k(grades: Sequence[int], total_relevant: int, k: int) -> float:
    """Fraction of gold spans covered by the top k.

    Denominator is the number of *gold spans*, not the number of graded hits:
    two hits covering the same gold span is one span found, not two.
    """
    if total_relevant == 0:
        return float("nan")
    found = sum(1 for g in grades[:k] if g > 0)
    return min(found, total_relevant) / total_relevant


def precision_at_k(grades: Sequence[int], k: int) -> float:
    """Fraction of the top k that is relevant.

    ``precision_at_k(g, 5)`` is the headline: Precision@5 predicts answer
    accuracy at r=0.98, which makes it the one retrieval number worth watching
    when there is only room for one.

    The denominator is ``k``, not ``len(grades[:k])``: a system returning three
    results of which two are relevant has not earned P@5 = 0.67.
    """
    if k == 0:
        return float("nan")
    return sum(1 for g in grades[:k] if g > 0) / k


def dcg(grades: Sequence[int], k: int) -> float:
    return float(sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k])))


def ndcg_at_k(grades: Sequence[int], ideal: Sequence[int], k: int) -> float:
    """Graded ranking quality. ``ideal`` is the gold weights, descending."""
    best = dcg(sorted(ideal, reverse=True), k)
    return dcg(grades, k) / best if best > 0 else float("nan")


def reciprocal_rank(grades: Sequence[int]) -> float:
    for i, g in enumerate(grades, start=1):
        if g > 0:
            return 1.0 / i
    return 0.0


def retrieval_failed(grades: Sequence[int], k: int) -> bool:
    """True when nothing relevant appears in the top k. The invariant-1 metric."""
    return not any(g > 0 for g in grades[:k])


@dataclass(slots=True)
class QueryScore:
    """Everything measured for one query in one arm.

    Per-query rather than aggregate-only, because the aggregate never tells you
    *which* queries a change broke -- and a change that improves the mean while
    breaking the structured questions is a change worth catching.
    """

    query_id: str
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    failed: bool = False
    #: Router correctness, scored against ``GoldenQuery.query_type``.
    route_correct: bool | None = None
    routed_path: str = ""
    #: End-to-end: did the system produce the right answer. ``None`` when no
    #: judge is configured -- distinct from ``False``, and conflating them would
    #: report an unjudged run as a total failure.
    correct: bool | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    hits_returned: int = 0
    error: str | None = None


@dataclass(slots=True)
class RunReport:
    """Aggregate for one arm. The row of an ablation table."""

    arm: str
    config_hash: str = ""
    manifest_id: str = ""
    n_queries: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    #: The headline. Share of queries with nothing relevant in the top
    #: ``failure_k``.
    retrieval_failure_rate: float = 0.0
    route_accuracy: float | None = None
    correctness: float | None = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    cost_per_query_usd: float = 0.0
    #: Same metrics sliced by query type. Aggregates hide the invariant-5
    #: failure mode entirely: structured questions are a minority of most sets,
    #: so routing them all to vector search costs a couple of points overall and
    #: is invisible -- while being catastrophic for that slice.
    by_query_type: dict[str, dict[str, float]] = field(default_factory=dict)
    errors: int = 0
    scores: list[QueryScore] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def headline(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "p@5": self.precision.get(5),
            "recall@20": self.recall.get(20),
            "ndcg@10": self.ndcg.get(10),
            "fail_rate": self.retrieval_failure_rate,
            "correct": self.correctness,
            "p50_ms": self.latency_p50_ms,
            "p95_ms": self.latency_p95_ms,
            "usd/q": self.cost_per_query_usd,
        }


class Judge(Protocol):
    """Decides end-to-end correctness for one query.

    A protocol, not an implementation, because the right judge depends on the
    answer type. Exact match for numbers and dates -- cheaper, deterministic and
    strictly more reliable than an LLM. An LLM judge for prose, with its own
    fingerprint recorded, since changing the judge changes the number and a
    correctness score whose judge is unrecorded is not comparable to anything.
    """

    def judge(self, item: GoldenQuery, answer: str, hits: Sequence[Hit]) -> bool | None:
        """True, False, or **None when the item cannot be assessed**.

        Abstention is not a nicety. A judge that returns False for items it
        cannot read reports them as wrong, which understates every arm by the
        same amount and makes the absolute correctness figure meaningless while
        leaving the deltas intact -- the shape of error that survives review
        because the comparison still looks sensible.

        ``None`` propagates to ``QueryScore.correct``, and the aggregate
        averages over judged items only.
        """
        ...

    def fingerprint(self) -> str: ...

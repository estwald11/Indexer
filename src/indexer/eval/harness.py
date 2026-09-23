"""The evaluation harness contracts.

Three protocols and one report type. The implementations come next round; what
is fixed here is the shape, because the shape is what makes invariant 6
("nothing is optimized without a before/after number") cheap enough to actually
follow.

``Bootstrapper``
    Turns a new corpus into a golden set, so adopting the library on a new
    corpus does not start with weeks of manual labelling. Its output is
    explicitly unverified.

``EvalRunner``
    Runs a golden set against a built index and produces a ``RunReport``.

``AblationRunner``
    Builds and evaluates several arms and prints the delta table.

Costing an ablation honestly
----------------------------
An arm that changes an ingestion stage requires a rebuild; an arm that changes
only the query path does not. The runner must distinguish them, or every
ablation pays full ingestion cost and nobody runs ablations -- which is how
invariant 6 quietly stops being followed. ``AblationSpec.requires_rebuild`` is
derived from which config keys an arm overrides: anything under ``ingestion``,
``corpus`` or ``paths`` rebuilds; anything under ``query`` reuses the index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from indexer.core.document import ParsedDocument
from indexer.core.results import RetrievalResponse
from indexer.eval.golden import GoldenSet
from indexer.eval.metrics import RunReport
from indexer.eval.stats import Comparison, compare

__all__ = [
    "AblationResult",
    "AblationRunner",
    "Bootstrapper",
    "EvalRunner",
    "QueryEngineLike",
    "SanityVerdict",
    "requires_rebuild",
]

#: Config prefixes whose change invalidates a built index.
#:
#: ``paths`` is deliberately absent: the ablation runner rewrites ``paths.store``
#: per arm so that each distinct ingestion configuration gets its own index, and
#: treating that rewrite as a content change would force a rebuild for every arm
#: -- defeating the reuse it exists to enable. What is actually in an index is
#: decided by ``corpus`` and ``ingestion``, which is what the runner fingerprints.
_REBUILD_PREFIXES = ("corpus", "ingestion", "cache")


def requires_rebuild(overrides: Mapping[str, Any]) -> bool:
    """Whether an arm's overrides touch ingestion. Decides rebuild vs reuse."""
    return any(k.split(".", 1)[0] in _REBUILD_PREFIXES for k in overrides)


class QueryEngineLike(Protocol):
    """The minimum the harness needs from a built system.

    Deliberately narrow: the harness must be able to evaluate anything that
    answers queries, including a baseline someone wants to compare against that
    was not built with this library at all. A wider interface would make
    "is our pipeline better than the thing we already have?" unanswerable, and
    that is usually the first question asked.
    """

    def query(self, text: str, *, top_k: int = 20) -> RetrievalResponse: ...


class Bootstrapper(Protocol):
    """Generates a golden set from a corpus.

    Contract
        Sample units, generate questions answerable *from that unit alone*,
        keep the source unit's span as gold, and filter aggressively.

    Must preserve
        *Answerability.* A generated question must be answerable from its source
        span. Verified by re-asking the generator with only that span.

        *Specificity.* A question answerable from a hundred other units is not
        an evaluation item; it is noise that makes every arm look identical. The
        standard filter is to run the question against a cheap lexical baseline
        and drop it when the source unit does not come out near the top -- items
        no system can distinguish tell you nothing about the systems.

        *Honest labelling.* Output is ``GoldOrigin.BOOTSTRAP``. Nothing marks it
        verified except a human.

        *Type coverage.* A set of only factual lookups cannot measure the
        router, and the router is where invariant 5 lives. The bootstrapper
        targets a distribution across ``QueryType``, generating structured and
        temporal questions from extracted fields rather than from prose.
    """

    def bootstrap(
        self, documents: Sequence[ParsedDocument], *, target_size: int = 200
    ) -> GoldenSet: ...

    def fingerprint(self) -> str: ...


class EvalRunner(Protocol):
    """Runs a golden set against a system and scores it."""

    def run(
        self, engine: QueryEngineLike, golden: GoldenSet, *, arm: str = "baseline"
    ) -> RunReport: ...


@dataclass(slots=True)
class SanityVerdict:
    """Result of one published-expectation check.

    The brief names two: contextualisation should cut retrieval failures by
    roughly a third, reranking should roughly halve them again. If it does not,
    something is wired wrong -- investigate before proceeding.

    The most common causes, in the order worth checking:
        1. The lexical index is indexing ``unit.text`` rather than
           ``indexing_text()``, so only the dense half sees the context.
        2. Gold spans no longer line up with the corpus (the parser changed and
           the canonical text shifted), so every arm scores badly and equally.
        3. The reranker is being fed ``input_top_k`` smaller than the depth the
           failures live at -- reranking a top-10 cannot fix a miss at 15.
        4. The golden set is too easy: if the baseline already fails on 1% of
           queries, there is no headroom and the deltas are noise.
    """

    name: str
    baseline_arm: str
    treatment_arm: str
    metric: str
    baseline_value: float
    treatment_value: float
    observed_reduction: float
    expected_reduction: float
    tolerance: float
    passed: bool
    diagnosis: str = ""

    def render(self) -> str:
        mark = "PASS" if self.passed else "INVESTIGATE"
        return (
            f"[{mark}] {self.name}: {self.metric} "
            f"{self.baseline_value:.3f} -> {self.treatment_value:.3f} "
            f"({self.observed_reduction:+.0%}, expected {-self.expected_reduction:.0%} "
            f"±{self.tolerance:.0%})"
            + (f"\n         {self.diagnosis}" if self.diagnosis and not self.passed else "")
        )


@dataclass(slots=True)
class AblationResult:
    """The delta table: one report per arm, plus the sanity verdicts."""

    reports: list[RunReport] = field(default_factory=list)
    verdicts: list[SanityVerdict] = field(default_factory=list)
    baseline_arm: str = "baseline"
    #: Wall time and spend for the ablation run itself. Ablations that cost more
    #: than the optimisation they justify are how invariant 6 dies in practice.
    total_cost_usd: float = 0.0
    total_wall_s: float = 0.0

    def delta_table(self) -> str:
        """Fixed-width table, deltas against the baseline arm.

        Plain text on purpose: this goes in a commit message, a PR body and a
        terminal, and it has to survive all three.
        """
        if not self.reports:
            return "(no arms)"
        base = next((r for r in self.reports if r.arm == self.baseline_arm), self.reports[0])

        # (header, width, accessor, decimals, lower_is_better | None)
        # `lower_is_better is None` means the column carries no delta: latency
        # and cost are reported absolutely because "slower" is not a regression
        # when it buys accuracy -- that trade is the reader's to make.
        cols: list[tuple[str, int, Any, int, bool | None]] = [
            ("arm", 24, lambda r: r.arm, 0, None),
            ("P@5", 16, lambda r: r.precision.get(5), 3, False),
            ("R@20", 16, lambda r: r.recall.get(20), 3, False),
            ("nDCG@10", 16, lambda r: r.ndcg.get(10), 3, False),
            ("fail%", 16, lambda r: r.retrieval_failure_rate, 3, True),
            ("correct", 16, lambda r: r.correctness, 3, False),
            ("p50ms", 8, lambda r: r.latency_p50_ms, 0, None),
            ("p95ms", 8, lambda r: r.latency_p95_ms, 0, None),
            ("usd/q", 9, lambda r: r.cost_per_query_usd, 5, None),
            # Errors are a column, not a footnote. Metrics are averaged over
            # the queries that ran, so an arm where most queries raised reports
            # healthy numbers over its survivors and looks like a good result.
            ("err", 6, lambda r: float(r.errors), 0, True),
        ]
        head = "".join(n.ljust(w) for n, w, _, _, _ in cols)
        lines = [head, "-" * len(head)]
        for r in self.reports:
            row = ""
            for name, width, get, dp, lower_better in cols:
                v = get(r)
                if name == "arm":
                    row += str(v)[: width - 1].ljust(width)
                    continue
                if v is None:
                    row += "-".ljust(width)
                    continue
                cell = f"{v:.{dp}f}"
                if lower_better is not None and r is not base:
                    bv = get(base)
                    if bv is not None:
                        d = v - bv
                        # Mark the direction explicitly rather than leaving the
                        # reader to remember that lower is better for one column
                        # and worse for the rest.
                        mark = "" if d == 0 else (">" if (d < 0) == lower_better else "<")
                        cell += f" ({d:+.{dp}f}){mark}"
                row += cell.ljust(width)
            lines.append(row.rstrip())
        broken = [r for r in self.reports if r.errors]
        if broken:
            lines.append("")
            for r in broken:
                share = r.errors / r.n_queries if r.n_queries else 1.0
                lines.append(
                    f"!! {r.arm}: {r.errors}/{r.n_queries} queries raised ({share:.0%}). "
                    f"Its metrics are averaged over the {r.n_queries - r.errors} that ran "
                    f"and are not comparable to the other arms."
                )
        significance = self.significance()
        if significance:
            lines.append("")
            lines.append(
                f"paired against {base.arm}, per query: bootstrap 95% interval for graded "
                f"metrics, McNemar for pass/fail ('*' = significant at 5%)"
            )
            for arm, comparisons in significance.items():
                lines.append(f"  {arm}")
                lines.extend(f"    {c.render()}" for c in comparisons)
        if self.verdicts:
            lines.append("")
            lines.extend(v.render() for v in self.verdicts)
        lines.append("")
        lines.append(
            f"{len(self.reports)} arms, {self.total_wall_s:.0f}s, ${self.total_cost_usd:.2f}"
            f"   ('>' better than {base.arm}, '<' worse)"
        )
        return "\n".join(lines)

    def significance(self) -> dict[str, list[Comparison]]:
        """Each arm against the baseline, per query, with an interval or a
        p-value per metric. A delta without one does not say whether it would
        survive the next golden set."""
        base = next((r for r in self.reports if r.arm == self.baseline_arm), None)
        if base is None or not base.scores:
            return {}
        return {r.arm: compare(base, r) for r in self.reports if r is not base and r.scores}


class AblationRunner(Protocol):
    """Builds and evaluates a set of arms, then prints the delta table."""

    def run(self, config_path: str, golden: GoldenSet) -> AblationResult: ...

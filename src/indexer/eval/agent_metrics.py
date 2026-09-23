"""Scoring what the agent is given, not only the ranking underneath it.

An agent reads the ``search`` tool's answer, not a ranked list: passages
clipped to a snippet, deduplicated and capped per document by the shape
policy, rows for the questions routed to the structured path. Retrieval
metrics score the list before any of that happens. These score the answer, per
golden query:

``found``          The passage that answers the question is among the results
                   the tool returned -- by the retrieval matcher's rule, span
                   overlap with the gold span.
``cited``          Every citation resolves: the passage exists, and its text is
                   what the result showed. A citation an agent cannot follow
                   back is a claim it cannot support.
``rows``           A question labelled structured, numeric or temporal was
                   answered with rows, not passages.
``payload_chars``  What the answer costs the agent's context: the size of the
                   JSON the model reads. A shape policy that halves it at the
                   same ``found`` rate is worth having.

The tools are duck-typed -- anything with ``search(query, top_k=...)`` and
``expand(unit_id, before=0, after=0)`` returning the same shapes -- so the
harness can score a tool set this library did not build.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from indexer.core.query import QueryType
from indexer.eval.golden import GoldenQuery, GoldenSet

__all__ = ["AgentToolReport", "AgentToolScore", "ToolsLike", "evaluate_tools"]

_STRUCTURED = (QueryType.STRUCTURED, QueryType.NUMERIC, QueryType.TEMPORAL)


class ToolsLike(Protocol):
    def search(self, query: str, *, top_k: int = 8) -> dict[str, Any]: ...

    def expand(self, unit_id: str, *, before: int = 1, after: int = 1) -> dict[str, Any]: ...


@dataclass(slots=True)
class AgentToolScore:
    query_id: str
    found: bool | None = None
    cited: bool = True
    rows: bool | None = None
    payload_chars: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class AgentToolReport:
    n_queries: int = 0
    #: Share of passage questions whose answer the tool returned.
    found_rate: float | None = None
    #: Share of results whose citation resolves to the text shown.
    citation_validity: float = 1.0
    #: Share of structured questions answered with rows.
    rows_rate: float | None = None
    payload_chars_p50: float = 0.0
    payload_chars_p95: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    errors: int = 0
    scores: list[AgentToolScore] = field(default_factory=list)

    def render(self) -> str:
        def pct(v: float | None) -> str:
            return "-" if v is None else f"{v:.1%}"

        return (
            f"agent tools over {self.n_queries} queries: found {pct(self.found_rate)}, "
            f"citations valid {self.citation_validity:.1%}, structured as rows "
            f"{pct(self.rows_rate)}, payload p50 {self.payload_chars_p50:.0f} / p95 "
            f"{self.payload_chars_p95:.0f} chars, latency p50 {self.latency_p50_ms:.0f} ms"
            + (f", {self.errors} errors" if self.errors else "")
        )


def evaluate_tools(
    tools: ToolsLike,
    golden: GoldenSet,
    *,
    top_k: int = 8,
    min_overlap: float = 0.5,
) -> AgentToolReport:
    scores = [_score(tools, item, top_k, min_overlap) for item in golden]
    ok = [s for s in scores if s.error is None]
    found = [s.found for s in ok if s.found is not None]
    rows = [s.rows for s in ok if s.rows is not None]
    sizes = sorted(float(s.payload_chars) for s in ok)
    times = sorted(s.latency_ms for s in ok)
    return AgentToolReport(
        n_queries=len(scores),
        found_rate=sum(found) / len(found) if found else None,
        citation_validity=sum(s.cited for s in ok) / len(ok) if ok else 1.0,
        rows_rate=sum(rows) / len(rows) if rows else None,
        payload_chars_p50=_pct(sizes, 50),
        payload_chars_p95=_pct(sizes, 95),
        latency_p50_ms=_pct(times, 50),
        latency_p95_ms=_pct(times, 95),
        errors=len(scores) - len(ok),
        scores=scores,
    )


def _score(tools: ToolsLike, item: GoldenQuery, top_k: int, min_overlap: float) -> AgentToolScore:
    s = AgentToolScore(query_id=item.id)
    t0 = time.perf_counter()
    try:
        answer = tools.search(item.query, top_k=top_k)
    except Exception as exc:
        s.error = f"{type(exc).__name__}: {exc}"
        return s
    s.latency_ms = (time.perf_counter() - t0) * 1000
    s.payload_chars = len(json.dumps(answer, ensure_ascii=False, default=str))
    results: Sequence[Mapping[str, Any]] = answer.get("results") or []
    if item.query_type in _STRUCTURED:
        s.rows = "rows" in answer
    if item.relevant:
        s.found = any(
            _matches(r, g.document_id, g.span, min_overlap) for r in results for g in item.relevant
        )
    s.cited = all(_resolves(tools, r) for r in results)
    return s


def _matches(result: Mapping[str, Any], document_id: str, gold: Any, min_overlap: float) -> bool:
    cite = result.get("citation") or {}
    if str(cite.get("document_id")) != str(document_id):
        return False
    start, end = cite.get("span", (0, 0))
    if gold.length == 0:
        return bool(start <= gold.start and gold.end <= end)
    covered = max(0, min(end, gold.end) - max(start, gold.start))
    return bool(covered / gold.length >= min_overlap)


def _resolves(tools: ToolsLike, result: Mapping[str, Any]) -> bool:
    cite = result.get("citation") or {}
    unit_id = cite.get("unit_id") or result.get("unit_id")
    if not unit_id:
        return False
    try:
        passage = tools.expand(str(unit_id), before=0, after=0).get("passage") or {}
    except Exception:
        return False
    shown = str(result.get("text", "")).removesuffix("...")
    return passage.get("citation", {}).get("span") == cite.get("span") and str(
        passage.get("text", "")
    ).removesuffix("...").startswith(shown[:200])


def _pct(values: Sequence[float], p: int) -> float:
    if not values:
        return 0.0
    i = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[i]

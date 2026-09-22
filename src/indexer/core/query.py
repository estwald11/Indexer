"""Queries and routing decisions.

`route` is the stage invariant 5 lives in. The contract has three parts:

1.  **At least three paths.** ``STRUCTURED`` answers from extracted fields
    without touching a vector index. ``LOOKUP`` takes one retrieval pass.
    ``ITERATIVE`` takes a bounded loop. ``RoutePath`` is an open string, not a
    closed enum, so a corpus that needs a fourth path adds one in config.

2.  **A step budget on every decision.** An unbounded loop is a latency and cost
    incident waiting to happen. ``LOOKUP`` carries budget 1 and the frame
    enforces it, so "one pass" is a guarantee rather than a convention.

3.  **Every decision is logged.** Including the non-decisions: when routing is
    disabled, the pipeline still records a ``RouteDecision`` with
    ``reason="stage_disabled"``. Without that, an ablation run has a hole in its
    trace exactly where the comparison needs data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from indexer.core.ids import ContentHash, hash_text
from indexer.core.predicate import Predicate, StructuredQuery

__all__ = ["Query", "QueryType", "RouteDecision", "RoutePath", "RouteTarget"]


class QueryType(StrEnum):
    """What kind of question this is. The golden set labels it; the router
    predicts it; the harness scores the router against the label."""

    FACTUAL = "factual"
    STRUCTURED = "structured"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


class RoutePath(StrEnum):
    """The three mandatory paths. Open for extension by string value."""

    STRUCTURED = "structured"
    LOOKUP = "lookup"
    ITERATIVE = "iterative"


@dataclass(frozen=True, slots=True)
class Query:
    """A question, plus whatever the caller already knows about it.

    ``filters`` is caller-supplied scoping (tenant, matter, date range) and is
    always applied; it is not the router's business to second-guess it. Filters
    the *router* infers from the text land on the decision instead, so that a
    trace can tell "the user asked for 2023" apart from "the router guessed
    2023" -- which matters when precision drops and someone has to find out why.
    """

    text: str
    filters: Predicate | None = None
    top_k: int = 20
    #: Caller hint. A router may honour or override it, but must record which.
    hint_type: QueryType | None = None
    #: Conversation turns or prior context, when a caller has them. The frame
    #: does not manage dialogue -- that is an agent concern and out of scope --
    #: but it will not drop context a caller supplies.
    context: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: Who is asking: the user and groups the caller has authenticated. With
    #: access control on, only documents whose ACL names one of these (or
    #: none, if so configured) are visible -- on every path, structured
    #: included. ``None`` means "not stated", which access control refuses
    #: rather than reads as "everyone".
    principals: tuple[str, ...] | None = None

    @property
    def content_hash(self) -> ContentHash:
        return hash_text(self.text)


@dataclass(frozen=True, slots=True)
class RouteTarget:
    """One index to query, and how hard.

    Per-target ``top_k`` because a lexical index and a dense index do not earn
    their depth at the same rate, and fusion works better when each list is
    sized to its own precision curve.
    """

    index: str
    top_k: int = 50
    #: Filter to push down to this index. Usually the query filters conjoined
    #: with anything the router inferred.
    filters: Predicate | None = None
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The router's output. Always produced, always logged, never silent.

    ``structured_query`` is populated iff ``path is STRUCTURED``; the query
    pipeline refuses the combination otherwise, because a structured route with
    nothing to ask the structured index would silently fall through to vector
    search and quietly violate invariant 5.
    """

    path: RoutePath | str
    targets: tuple[RouteTarget, ...] = field(default_factory=tuple)
    #: Hard cap on retrieval rounds. 1 for LOOKUP. Enforced by the frame.
    step_budget: int = 1
    structured_query: StructuredQuery | None = None
    query_type: QueryType = QueryType.UNKNOWN
    #: Sub-queries for the iterative path, or rewrites/expansions for lookup.
    sub_queries: tuple[str, ...] = field(default_factory=tuple)
    #: Filters the *router* inferred from the text, kept separate from the
    #: caller's so a trace can attribute a bad result to the right party.
    inferred_filters: Predicate | None = None
    confidence: float = 1.0
    #: Human-readable justification. Written to the query trace verbatim.
    reason: str = ""
    router: str = ""
    fingerprint: str = ""
    #: A standalone form of the question, when the router rewrote it -- a
    #: follow-up ("e quella di marzo?") made self-contained from the caller's
    #: context. Retrieval uses it; the trace keeps both.
    rewritten_query: str | None = None

    def __post_init__(self) -> None:
        if self.step_budget < 1:
            raise ValueError("step_budget must be at least 1")
        if str(self.path) == RoutePath.LOOKUP and self.step_budget != 1:
            raise ValueError("the LOOKUP path is one retrieval pass by definition")
        if str(self.path) == RoutePath.STRUCTURED and self.structured_query is None:
            raise ValueError(
                "STRUCTURED route without a structured_query would fall through to "
                "vector search; emit a query or choose another path"
            )

    def as_log_record(self) -> dict[str, Any]:
        """Flat, JSON-safe form for the decision log."""
        return {
            "path": str(self.path),
            "query_type": str(self.query_type),
            "targets": [t.index for t in self.targets],
            "top_k": {t.index: t.top_k for t in self.targets},
            "step_budget": self.step_budget,
            "sub_queries": list(self.sub_queries),
            "confidence": self.confidence,
            "reason": self.reason,
            "router": self.router,
            "fingerprint": self.fingerprint,
            "has_structured_query": self.structured_query is not None,
            "has_inferred_filters": self.inferred_filters is not None,
            "rewritten_query": self.rewritten_query,
        }

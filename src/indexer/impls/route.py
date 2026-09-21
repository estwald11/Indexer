"""Routers: rules-based, and a passthrough.

The rules router is deliberately the reference implementation rather than a
placeholder for an LLM one. Reasons, in order of weight:

1. It is ~0 latency and ~0 cost on the path that carries most traffic.
2. It is the control arm. "The LLM router is worth it" is a claim that needs a
   cheap baseline to be measured against, and a rules router is a strong one --
   the signals that distinguish a numeric question from a prose one are mostly
   lexical.
3. It is deterministic, so a routing regression is attributable to a rule rather
   than to sampling.

Every decision it makes is logged with the rule that fired, which is what makes
"40% of traffic is taking the iterative path" a diagnosable observation rather
than a mystery.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from indexer.core.predicate import (
    Compare,
    Exists,
    Op,
    Predicate,
    StructuredQuery,
    TextMatch,
    all_of,
)
from indexer.core.query import Query, QueryType, RouteDecision, RoutePath, RouteTarget
from indexer.core.registry import register
from indexer.core.stages import StageContext
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["PassthroughRouter", "RulesRouter"]

# Ordered most-specific first: the first rule that fires wins, and an aggregate
# question that also mentions a date is an aggregate question.
_AGGREGATE = re.compile(
    r"\b(how many|count of|number of|total|sum of|average|mean|median|"
    r"most|least|largest|smallest|highest|lowest|top \d+|list all|which \w+ (have|has|are))\b",
    re.I,
)
_TEMPORAL = re.compile(
    r"\b(before|after|since|until|between .{1,20} and|earlier than|later than|"
    r"in \d{4}|as of|expir\w+|effective|released? (in|on)|last (year|month|week))\b",
    re.I,
)
_NUMERIC = re.compile(
    r"(\b(greater|less|more|fewer|at least|at most|over|under|above|below)\s+than?\b"
    r"|[<>]=?\s*\d|\b\d+(\.\d+)?\s*(%|usd|eur|gbp|k|m|bn|million|billion)\b)",
    re.I,
)
_COMPARATIVE = re.compile(
    r"\b(compare|versus|vs\.?|difference between|better than|instead of|"
    r"trade-?offs? between)\b",
    re.I,
)
_MULTI_HOP = re.compile(
    r"\b(and (then|also)|as well as|both .{1,30} and|why does .{1,40} when|"
    r"how does .{1,30} affect|what causes|implications? of)\b",
    re.I,
)
_SUMMARY = re.compile(
    r"\b(summar\w+|overview of|explain|what is the purpose|walk me through)\b", re.I
)
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{4})\b")

#: Comparison words, longest first so "greater than or equal" beats "greater".
#: Temporal words are comparisons too -- without them every temporal question
#: collapses to equality and matches nothing.
_OPERATORS: dict[str, Op] = {
    "greater than or equal to": Op.GTE,
    "less than or equal to": Op.LTE,
    "at least": Op.GTE,
    "no less than": Op.GTE,
    "at most": Op.LTE,
    "no more than": Op.LTE,
    "greater than": Op.GT,
    "more than": Op.GT,
    "larger than": Op.GT,
    "later than": Op.GT,
    "less than": Op.LT,
    "fewer than": Op.LT,
    "smaller than": Op.LT,
    "earlier than": Op.LT,
    "not after": Op.LTE,
    "not before": Op.GTE,
    "before": Op.LT,
    "after": Op.GT,
    "since": Op.GTE,
    "until": Op.LTE,
    "above": Op.GT,
    "below": Op.LT,
    "over": Op.GT,
    "under": Op.LT,
    ">=": Op.GTE,
    "<=": Op.LTE,
    ">": Op.GT,
    "<": Op.LT,
    "=": Op.EQ,
}
_OP_WORDS = "|".join(re.escape(k) for k in sorted(_OPERATORS, key=len, reverse=True))
#: Words that follow a field name without being its value.
_FILLER = frozenset(
    {"is", "are", "of", "the", "a", "an", "and", "or", "recorded", "values", "value"}
)


def _op_from(token: str | None, default: Op) -> Op:
    return _OPERATORS.get((token or "").strip().lower(), default)


def _fields_in(text: str, lexicon: Sequence[str]) -> list[str]:
    """Fields the text names, longest match first and no overlaps.

    "version major" contains "version", so a naive substring scan yields both
    and conjoins a predicate on each -- and a unit carrying `version_major` but
    no `version` then matches neither.
    """
    low = text.lower()
    taken: list[tuple[int, int]] = []
    found: list[str] = []
    for f in sorted(lexicon, key=len, reverse=True):
        needle = f.replace("_", " ").lower()
        i = low.find(needle)
        if i < 0:
            continue
        j = i + len(needle)
        if any(i < b and a < j for a, b in taken):
            continue
        taken.append((i, j))
        found.append(f)
    return [f for _, f in sorted(zip([t[0] for t in taken], found, strict=True))]


@dataclass(frozen=True, slots=True)
class RulesRouterParams:
    #: Index names per path. Supplied by the assembler from ``route.paths``.
    paths: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_top_k: int = 50
    #: Field names the structured index knows about. A question naming one is a
    #: strong structured signal, and this is how a corpus teaches the router its
    #: own vocabulary without a code change.
    field_lexicon: list[str] = field(default_factory=list)
    #: Declared type per field (str/int/float/bool/date). Supplied from the
    #: extraction config. Without it the router guesses from the value's
    #: spelling, and a dotted version string reads as a float.
    field_types: dict[str, str] = field(default_factory=dict)
    iterative_min_words: int = 14
    enable_structured: bool = True


@register(
    "route",
    "rules",
    version="1",
    params_model=dataclass_params(RulesRouterParams),
    summary="Regex and field-lexicon rules. Zero latency, deterministic, a strong baseline.",
)
def _make_rules(params: dict[str, Any], **_: Any) -> RulesRouter:
    return RulesRouter(params)


class RulesRouter(StageImpl):
    STAGE, IMPL, VERSION = "route", "rules", "1"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.paths: dict[str, dict[str, Any]] = params.get("paths", {})
        self.lexicon = [w.lower() for w in params.get("field_lexicon", [])]
        self.field_types: dict[str, str] = {
            k: str(v).lower() for k, v in (params.get("field_types") or {}).items()
        }
        self.default_top_k = int(params.get("default_top_k", 50))
        self.iterative_min_words = int(params.get("iterative_min_words", 14))
        self.enable_structured = bool(params.get("enable_structured", True))

    def route(self, query: Query, ctx: StageContext) -> RouteDecision:
        text = query.text
        qtype, reason = self._classify(text)

        wants_structured = qtype in (
            QueryType.STRUCTURED,
            QueryType.NUMERIC,
            QueryType.TEMPORAL,
        )
        if wants_structured and self.enable_structured:
            sq = self._structured_query(text, qtype)
            if sq is not None and self._targets("structured"):
                return RouteDecision(
                    path=RoutePath.STRUCTURED,
                    targets=self._targets("structured"),
                    step_budget=1,
                    structured_query=sq,
                    query_type=qtype,
                    confidence=0.8,
                    reason=reason,
                    router=self.IMPL,
                    fingerprint=self.fingerprint().key(),
                )
            # Nothing extractable from the text. Fall back to lookup, but
            # record *why* -- see below.
            reason = f"{reason}; no structured predicate extractable, falling back to lookup"
        elif wants_structured:
            # The question wanted the structured path and the configuration does
            # not offer one. An operator reading the decision log needs to see
            # that, not a line saying the question looked like prose: the fix is
            # in the config, and nothing else in the trace would point there.
            reason = f"{reason}; structured path unavailable (no structured index), using lookup"

        if qtype in (QueryType.MULTI_HOP, QueryType.COMPARATIVE) or (
            len(text.split()) >= self.iterative_min_words and qtype is QueryType.SUMMARY
        ):
            spec = self.paths.get("iterative", {})
            return RouteDecision(
                path=RoutePath.ITERATIVE,
                targets=self._targets("iterative"),
                step_budget=int(spec.get("step_budget", 3)),
                query_type=qtype,
                sub_queries=self._decompose(text),
                confidence=0.6,
                reason=reason,
                router=self.IMPL,
                fingerprint=self.fingerprint().key(),
            )

        return RouteDecision(
            path=RoutePath.LOOKUP,
            targets=self._targets("lookup"),
            step_budget=1,
            query_type=qtype,
            confidence=0.7,
            reason=reason,
            router=self.IMPL,
            fingerprint=self.fingerprint().key(),
        )

    # ------------------------------------------------------------- internals

    def _classify(self, text: str) -> tuple[QueryType, str]:
        mentions_field = [f for f in self.lexicon if f.replace("_", " ") in text.lower()]
        if _AGGREGATE.search(text) and (mentions_field or _NUMERIC.search(text)):
            return (
                QueryType.STRUCTURED,
                f"aggregate + field/numeric ({', '.join(mentions_field) or 'numeric'})",
            )
        if _NUMERIC.search(text):
            return QueryType.NUMERIC, "numeric comparison"
        if _TEMPORAL.search(text) and (mentions_field or _DATE.search(text)):
            return QueryType.TEMPORAL, "temporal constraint"
        if _COMPARATIVE.search(text):
            return QueryType.COMPARATIVE, "comparative"
        if _MULTI_HOP.search(text):
            return QueryType.MULTI_HOP, "multi-hop phrasing"
        if _SUMMARY.search(text):
            return QueryType.SUMMARY, "summary request"
        if mentions_field and _AGGREGATE.search(text):
            return QueryType.STRUCTURED, f"field mention: {mentions_field[0]}"
        return QueryType.FACTUAL, "no structural signal; factual lookup"

    def _structured_query(self, text: str, qtype: QueryType) -> StructuredQuery | None:
        """Extract a predicate from the text. Conservative by design.

        Returning ``None`` when nothing is confidently extractable is the right
        behaviour: a STRUCTURED route with a wrong predicate returns a confident
        wrong answer, which is worse than falling back to retrieval.

        Four things here are easy to get wrong, and all four were, until the
        structured slice of a real golden set showed well-formed questions
        returning nothing:

        *Dates are not numbers.* "release date before 2023-06-27" read as
        ``release_date == 2023`` compares a date column against an integer and
        matches nothing. ISO dates are parsed before numbers.

        *Temporal words are operators.* before, after, since, until carry the
        comparison; without them every temporal question collapsed to equality.

        *"greater than" is an operator too.* The lexicon knew ``>`` and "over"
        but not the words most people actually type.

        *The longest field name wins.* "version major" contains "version", so a
        substring match produced a predicate on both fields conjoined -- and a
        unit with a ``version_major`` but no ``version`` matched neither.
        """
        fields_named = _fields_in(text, self.lexicon)
        if not fields_named:
            return None

        clauses: list[Predicate] = []
        for f in fields_named:
            label = re.escape(f.replace("_", " "))
            window = rf"{label}\s*(?:is|of|=)?\s*({_OP_WORDS})?\s*"
            declared = self.field_types.get(f)

            if declared in (None, "date", "datetime"):
                m = re.search(window + r"(\d{4}-\d{2}-\d{2})", text, re.I)
                if m:
                    clauses.append(
                        Compare(
                            f, _op_from(m.group(1), default=Op.EQ), date.fromisoformat(m.group(2))
                        )
                    )
                    continue

            # A field declared textual takes a textual value, however numeric
            # the value looks. "version 1.0.0" is a string in every corpus that
            # has ever had a patch release, and reading it as 1.0 queries a
            # numeric column the value was never written to.
            if declared in (None, "int", "float", "bool"):
                m = re.search(window + r"(\d[\d,_]*(?:\.\d+)?)", text, re.I)
                if m:
                    raw = m.group(2).replace(",", "").replace("_", "")
                    value: Any = float(raw) if "." in raw else int(raw)
                    if declared == "int" and isinstance(value, float):
                        value = int(value)
                    clauses.append(Compare(f, _op_from(m.group(1), default=Op.EQ), value))
                    continue

            m = re.search(window + r"([A-Za-z0-9][\w.\-]{0,40})", text, re.I)
            if m and m.group(2).lower() not in _FILLER:
                clauses.append(
                    Compare(f, _op_from(m.group(1), default=Op.EQ), m.group(2))
                    if _op_from(m.group(1), default=Op.EQ) is not Op.EQ
                    else TextMatch(f, m.group(2), mode="exact")
                )
                continue
            clauses.append(Exists(f))

        if not clauses:
            return None
        return StructuredQuery(
            where=all_of(clauses),
            select=tuple(fields_named[:4]) or ("unit_id",),
            limit=50,
        )

    def _decompose(self, text: str) -> tuple[str, ...]:
        """Split a comparative or multi-hop question into retrievable parts."""
        parts = re.split(r"\b(?:versus|vs\.?|compared to|and then|as well as)\b", text, flags=re.I)
        cleaned = [p.strip(" ,?.") for p in parts if len(p.strip()) > 8]
        return tuple(cleaned) if len(cleaned) > 1 else (text,)

    def _targets(self, path: str) -> tuple[RouteTarget, ...]:
        spec = self.paths.get(path, {})
        top_k = spec.get("top_k", {})
        return tuple(
            RouteTarget(index=name, top_k=int(top_k.get(name, self.default_top_k)))
            for name in spec.get("targets", [])
        )


# --------------------------------------------------------------------------- #
# passthrough                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PassthroughRouterParams:
    targets: list[str] = field(default_factory=list)
    top_k: int = 50


@register(
    "route",
    "passthrough",
    version="1",
    params_model=dataclass_params(PassthroughRouterParams),
    summary="Everything takes LOOKUP against all indexes. The no-router ablation arm.",
)
def _make_passthrough_router(params: dict[str, Any], **_: Any) -> PassthroughRouter:
    return PassthroughRouter(params)


class PassthroughRouter(StageImpl):
    """One path for everything.

    Deliberately violates invariant 5, which is the point: the gap between this
    and the rules router *is* the measured value of routing, and without an arm
    that does the wrong thing there is no number.
    """

    STAGE, IMPL, VERSION = "route", "passthrough", "1"

    def route(self, query: Query, ctx: StageContext) -> RouteDecision:
        return RouteDecision(
            path=RoutePath.LOOKUP,
            targets=tuple(
                RouteTarget(index=n, top_k=int(self.param("top_k", 50)))
                for n in self.param("targets", [])
            ),
            step_budget=1,
            query_type=QueryType.UNKNOWN,
            confidence=0.0,
            reason="passthrough: no routing performed",
            router=self.IMPL,
            fingerprint=self.fingerprint().key(),
        )

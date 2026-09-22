"""A small predicate AST: the frame's only query language for structured facts.

Invariant 5 -- structured, numeric and temporal questions must never reach
vector search -- needs somewhere for those questions to go instead. That is the
structured index, and this is what it is asked in.

One AST serves two jobs deliberately:

*   the ``STRUCTURED`` route's whole query ("contracts expiring before March
    with value over 1M"), and
*   metadata filters attached to a dense or lexical search ("...within this
    matter, 2023 onwards").

Keeping them the same type means a router that extracts a date constraint does
not have to know which path will consume it, and an index that supports
filtering supports structured lookup for free.

It is deliberately smaller than SQL. It compiles cleanly to SQL ``WHERE``, to
Qdrant/LanceDB filters, and to a pure-Python evaluator for tests -- which is the
test of whether it is small enough. Joins, subqueries and arbitrary expressions
are out: a corpus that needs them needs a database, and the documented seam is
to register a structured index implementation that speaks one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias

from indexer.core.unit import FieldValue

__all__ = [
    "Aggregation",
    "AggregationOp",
    "And",
    "Compare",
    "Exists",
    "In",
    "Not",
    "Op",
    "Or",
    "Predicate",
    "StructuredQuery",
    "TextMatch",
    "evaluate",
]


class Op(StrEnum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"


@dataclass(frozen=True, slots=True)
class Compare:
    """``field <op> value``. The value's type must match the field's."""

    field: str
    op: Op
    value: FieldValue


@dataclass(frozen=True, slots=True)
class In:
    field: str
    values: tuple[FieldValue, ...]


@dataclass(frozen=True, slots=True)
class Exists:
    """Whether a field was extracted at all.

    Distinct from ``Compare(field, EQ, None)``: "no governing-law clause found"
    and "governing law is explicitly none" are different answers, and an
    extraction pipeline that cannot tell them apart will report the wrong one.
    """

    field: str
    present: bool = True


@dataclass(frozen=True, slots=True)
class TextMatch:
    """Substring or prefix match on a text field.

    Present because "party name contains Acme" is a structured question that
    would otherwise be pushed to vector search in violation of invariant 5.
    Not a full-text query -- that is the lexical index's job.
    """

    field: str
    value: str
    mode: str = "contains"  # contains | prefix | exact


@dataclass(frozen=True, slots=True)
class And:
    clauses: tuple[Predicate, ...]


@dataclass(frozen=True, slots=True)
class Or:
    clauses: tuple[Predicate, ...]


@dataclass(frozen=True, slots=True)
class Not:
    clause: Predicate


Predicate: TypeAlias = Compare | In | Exists | TextMatch | And | Or | Not


class AggregationOp(StrEnum):
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class Aggregation:
    op: AggregationOp
    field: str | None = None  # None only for COUNT
    #: COUNT only: count distinct values of ``field`` rather than rows --
    #: "how many suppliers", not "how many invoices from suppliers".
    distinct: bool = False


@dataclass(frozen=True, slots=True)
class StructuredQuery:
    """What the ``STRUCTURED`` route sends to the structured index.

    Returns rows, not a ranked list: "which contracts expire in Q1" has an
    answer set, not a relevance order. The response still carries unit ids so
    every row cites its source -- provenance holds on this path too.

    ``level`` says what a row *is*. ``"unit"`` (the default, and the historical
    behaviour) evaluates the predicate per unit and counts units. ``"document"``
    evaluates it over each document's fields together and counts documents --
    which is what "how many invoices over 1,000 euros" means, and what a
    predicate needs when the supplier is named on page one and the total on
    page three.
    """

    where: Predicate | None = None
    select: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    aggregations: tuple[Aggregation, ...] = ()
    order_by: tuple[tuple[str, bool], ...] = ()  # (field, descending)
    limit: int | None = None
    level: str = "unit"  # unit | document


def evaluate(pred: Predicate, fields: dict[str, Any]) -> bool:
    """Reference in-memory evaluation of a predicate.

    Not the production path -- real stores push predicates down. It exists so
    the AST has an executable definition: any store implementation must agree
    with this function, and the conformance tests check exactly that.

    A field may hold a list -- the groups an ACL grants, the VAT numbers a
    document mentions. Every comparison on a list is *existential*: it holds if
    it holds for some element, and an empty list is an absent field. So
    ``In("acl", principals)`` asks "does the caller hold any group this
    document grants", which is the question an access check asks.
    """
    match pred:
        case Compare(field=f, op=op, value=v):
            compared: Any = fields.get(f)
            if _is_list(compared):
                if not compared:
                    return _compare(f, None, op, v)
                return any(_compare(f, a, op, v) for a in compared)
            return _compare(f, compared, op, v)
        case In(field=f, values=vs):
            member: Any = fields.get(f)
            if _is_list(member):
                return any(a in vs for a in member)
            return member in vs
        case Exists(field=f, present=p):
            held: Any = fields.get(f)
            present = bool(held) if _is_list(held) else held is not None
            return present is p
        case TextMatch(field=f, value=v, mode=m):
            text: Any = fields.get(f)
            values = list(text) if _is_list(text) else [text]
            return any(_text_match(a, v, m) for a in values)
        case And(clauses=cs):
            return all(evaluate(c, fields) for c in cs)
        case Or(clauses=cs):
            return any(evaluate(c, fields) for c in cs)
        case Not(clause=c):
            return not evaluate(c, fields)
    raise TypeError(f"unknown predicate node {type(pred).__name__}")


def _is_list(v: Any) -> bool:
    return isinstance(v, (list, tuple))


def _compare(f: str, actual: Any, op: Op, v: Any) -> bool:
    if actual is None or v is None:
        return op is Op.EQ and actual is None and v is None
    try:
        match op:
            case Op.EQ:
                return bool(actual == v)
            case Op.NE:
                return bool(actual != v)
            case Op.LT:
                return bool(actual < v)
            case Op.LTE:
                return bool(actual <= v)
            case Op.GT:
                return bool(actual > v)
            case Op.GTE:
                return bool(actual >= v)
    except TypeError:
        # Comparing a date to a string is a config or extraction bug.
        # Returning False would hide it behind an empty result set.
        raise TypeError(
            f"field {f!r} holds {type(actual).__name__}, compared against "
            f"{type(v).__name__}; fix the extraction schema"
        ) from None
    raise ValueError(f"unknown operator {op!r}")


def _text_match(actual: Any, value: str, mode: str) -> bool:
    if not isinstance(actual, str):
        return False
    hay, needle = actual.casefold(), value.casefold()
    match mode:
        case "contains":
            return needle in hay
        case "prefix":
            return hay.startswith(needle)
        case "exact":
            return hay == needle
    raise ValueError(f"unknown TextMatch mode {mode!r}")


def all_of(clauses: Sequence[Predicate]) -> Predicate | None:
    """Conjoin, collapsing the 0- and 1-clause cases. Routers build filters up."""
    kept = tuple(clauses)
    if not kept:
        return None
    return kept[0] if len(kept) == 1 else And(kept)

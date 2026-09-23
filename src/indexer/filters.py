"""Filters written as data: ``{"field": ..., "op": ..., "value": ...}``.

The shape an agent or a model writes a condition in, turned into a
``Predicate`` the indexes evaluate. One reading for every writer -- the LLM
router, the agent tools -- so "importo gt 1000" means the same comparison
whoever wrote it, and a value is coerced to its field's declared type the way
the rules router reads a question: "1.250,00" is a number, "30/06/2025" a date.

A condition that cannot mean anything -- an unknown field, ``gt`` on a text
field, "molti" as an amount -- raises ``FilterError`` rather than becoming a
predicate that silently matches nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from indexer.core.errors import IndexerError
from indexer.core.predicate import (
    Compare,
    Exists,
    In,
    Op,
    Predicate,
    TextMatch,
    all_of,
)
from indexer.normalize import parse_bool, parse_date, parse_number

__all__ = ["OPS", "FilterError", "build_clause", "coerce", "parse_filters"]

OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "contains", "prefix", "in", "exists", "missing")
_NUMERIC = ("int", "float")


class FilterError(IndexerError, ValueError):
    """A condition that cannot be evaluated as written."""


def kind_of(value: Any) -> str:
    """The type a value implies when its field declares none."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "float"
    if isinstance(value, str) and parse_date(value) is not None:
        return "date"
    return "str"


def coerce(name: str, kind: str, value: Any, *, locale: str = "auto") -> Any:
    """``value`` as a ``kind``, read the way people write it."""
    if value is None:
        raise FilterError(f"no value for {name}")
    out: Any = None
    if kind in ("date", "datetime"):
        out = parse_date(str(value))
    elif kind in _NUMERIC:
        number: float | None
        if isinstance(value, bool):
            number = None
        elif isinstance(value, int | float):
            number = float(value)
        else:
            number = parse_number(str(value), locale)
        if number is not None and (kind == "float" or number == int(number)):
            out = int(number) if kind == "int" else number
    elif kind == "bool":
        out = value if isinstance(value, bool) else parse_bool(str(value))
    else:
        out = str(value).strip() or None
    if out is None:
        raise FilterError(f"{value!r} is not a {kind or 'value'} for {name}")
    return out


def build_clause(
    name: str,
    op: str,
    value: Any,
    kind: str = "",
    *,
    locale: str = "auto",
) -> Predicate:
    """One condition. ``kind`` is the field's declared type; empty means infer
    it from the value."""
    if op == "exists":
        return Exists(name)
    if op == "missing":
        return Exists(name, present=False)
    if op not in OPS:
        raise FilterError(f"unknown operator {op!r}; use one of {', '.join(OPS)}")
    if op == "in":
        values = value if isinstance(value, list | tuple) else [value]
        if not values:
            raise FilterError(f"`in` on {name} needs at least one value")
        k = kind or kind_of(values[0])
        return In(name, tuple(coerce(name, k, v, locale=locale) for v in values))
    kind = kind or kind_of(value)
    typed = coerce(name, kind, value, locale=locale)
    if op in ("contains", "prefix"):
        if kind != "str":
            raise FilterError(f"{op} needs a text field; {name} is {kind}")
        return TextMatch(name, str(typed), mode=op)
    if kind == "str" and op == "eq":
        # As the rules router reads it: equality on text ignores case.
        return TextMatch(name, str(typed), mode="exact")
    if kind in ("str", "bool") and op not in ("eq", "ne"):
        raise FilterError(f"{op} needs a number or a date; {name} is {kind}")
    return Compare(name, Op(op), typed)


def parse_filters(
    filters: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    types: Mapping[str, str],
    *,
    known_only: bool = True,
    locale: str = "auto",
) -> Predicate | None:
    """Conditions, conjoined. Either ``{field: value, ...}`` for equalities or a
    list of ``{"field", "op", "value"}``. With ``known_only`` a field outside
    ``types`` is an error: a typo must not become a filter that matches nothing."""
    if not filters:
        return None
    items: list[Mapping[str, Any]]
    if isinstance(filters, Mapping):
        items = [{"field": k, "op": "eq", "value": v} for k, v in filters.items()]
    else:
        items = list(filters)
    clauses: list[Predicate] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise FilterError(f"a filter is a mapping with field, op and value, not {item!r}")
        name = str(item.get("field", ""))
        if not name or (known_only and name not in types):
            known = ", ".join(sorted(types)) or "(none)"
            raise FilterError(f"unknown field {name!r}; known fields: {known}")
        clauses.append(
            build_clause(
                name,
                str(item.get("op", "eq")),
                item.get("value"),
                types.get(name, ""),
                locale=locale,
            )
        )
    return all_of(clauses)

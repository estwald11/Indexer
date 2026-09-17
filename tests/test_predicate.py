"""Tests for the predicate AST -- the language invariant 5 depends on.

``evaluate`` is the AST's executable definition. Every store implementation must
agree with it, so these are also the conformance tests a SQLite or Qdrant filter
compiler will be checked against.
"""

from __future__ import annotations

from datetime import date

import pytest

from indexer.core.predicate import (
    And,
    Compare,
    Exists,
    In,
    Not,
    Op,
    Or,
    TextMatch,
    all_of,
    evaluate,
)

FIELDS = {
    "counterparty": "Acme Industrial GmbH",
    "contract_value": 1_250_000.0,
    "effective_date": date(2023, 4, 1),
    "governing_law": None,
    "renewable": True,
}


class TestComparison:
    def test_numeric(self) -> None:
        assert evaluate(Compare("contract_value", Op.GT, 1_000_000), FIELDS)
        assert not evaluate(Compare("contract_value", Op.GT, 2_000_000), FIELDS)

    def test_temporal(self) -> None:
        assert evaluate(Compare("effective_date", Op.LT, date(2024, 1, 1)), FIELDS)
        assert not evaluate(Compare("effective_date", Op.LT, date(2023, 1, 1)), FIELDS)

    def test_missing_field_is_not_a_match(self) -> None:
        assert not evaluate(Compare("nonexistent", Op.EQ, "x"), FIELDS)

    def test_type_mismatch_raises_rather_than_returning_empty(self) -> None:
        """An empty result set would hide an extraction-schema bug indefinitely."""
        with pytest.raises(TypeError, match="fix the extraction schema"):
            evaluate(Compare("effective_date", Op.GT, "last tuesday"), FIELDS)


class TestExists:
    def test_distinguishes_absent_from_explicitly_none(self) -> None:
        """'No clause found' and 'explicitly none' are different answers."""
        assert not evaluate(Exists("governing_law"), FIELDS)
        assert evaluate(Exists("governing_law", present=False), FIELDS)
        assert evaluate(Exists("counterparty"), FIELDS)


class TestTextMatch:
    def test_contains_is_case_insensitive(self) -> None:
        assert evaluate(TextMatch("counterparty", "acme"), FIELDS)

    def test_prefix_and_exact(self) -> None:
        assert evaluate(TextMatch("counterparty", "Acme", mode="prefix"), FIELDS)
        assert not evaluate(TextMatch("counterparty", "Acme", mode="exact"), FIELDS)

    def test_non_text_field_does_not_match(self) -> None:
        assert not evaluate(TextMatch("contract_value", "1250000"), FIELDS)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown TextMatch mode"):
            evaluate(TextMatch("counterparty", "x", mode="fuzzy"), FIELDS)


class TestBoolean:
    def test_and_or_not(self) -> None:
        q = And(
            (
                Compare("contract_value", Op.GTE, 1_000_000),
                Or((TextMatch("counterparty", "acme"), TextMatch("counterparty", "globex"))),
                Not(Exists("governing_law")),
            )
        )
        assert evaluate(q, FIELDS)

    def test_in(self) -> None:
        assert evaluate(In("counterparty", ("Acme Industrial GmbH", "Globex")), FIELDS)
        assert not evaluate(In("counterparty", ("Globex",)), FIELDS)


class TestAllOf:
    def test_collapses_trivial_cases(self) -> None:
        assert all_of([]) is None
        single = Compare("a", Op.EQ, 1)
        assert all_of([single]) is single
        assert isinstance(all_of([single, Compare("b", Op.EQ, 2)]), And)

"""Tests for the rules router's predicate extraction.

Invariant 5's path is only as good as the predicate the router builds. A
STRUCTURED decision carrying a wrong predicate returns a confident wrong answer
-- or, more often, an empty one that scores as a retrieval failure and gets
blamed on the retriever.

Every case here failed before the structured slice of a real golden set made it
visible: well-formed questions compiling to predicates that matched nothing.
"""

from __future__ import annotations

from datetime import date

import pytest

from indexer.core.predicate import Compare, Exists, Op, TextMatch, evaluate
from indexer.core.query import Query, QueryType, RoutePath
from indexer.impls.route import RulesRouter

LEXICON = ["package", "release_date", "version", "version_major"]

ROWS = [
    {"package": "flask", "version_major": 3, "release_date": date(2025, 11, 2)},
    {"package": "click", "version_major": 8, "release_date": date(2022, 7, 30)},
    {"package": "attrs", "version_major": 25, "release_date": date(2024, 5, 1)},
]


@pytest.fixture
def router() -> RulesRouter:
    return RulesRouter(
        {
            "field_lexicon": LEXICON,
            "paths": {
                "structured": {"targets": ["fields"]},
                "lookup": {"targets": ["lexical", "dense"]},
                "iterative": {"targets": ["lexical", "dense"], "step_budget": 3},
            },
        }
    )


class TestTypedExtraction:
    def test_iso_dates_are_dates_not_numbers(self, router: RulesRouter) -> None:
        """`release_date == 2023` compares a date column against an integer and
        matches nothing, however well-formed the question was."""
        sq = router._structured_query("which entries have release date before 2023-06-27", None)
        assert sq is not None
        assert sq.where == Compare("release_date", Op.LT, date(2023, 6, 27))

    def test_temporal_words_are_operators(self, router: RulesRouter) -> None:
        for phrase, op in [
            ("before 2020-01-01", Op.LT),
            ("after 2020-01-01", Op.GT),
            ("since 2020-01-01", Op.GTE),
            ("until 2020-01-01", Op.LTE),
            ("earlier than 2020-01-01", Op.LT),
        ]:
            sq = router._structured_query(f"which entries have release date {phrase}", None)
            assert sq is not None, phrase
            assert sq.where == Compare("release_date", op, date(2020, 1, 1)), phrase

    def test_comparison_words_people_actually_type(self, router: RulesRouter) -> None:
        for phrase, op in [
            ("greater than 3", Op.GT),
            ("more than 3", Op.GT),
            ("less than 3", Op.LT),
            ("fewer than 3", Op.LT),
            ("at least 3", Op.GTE),
            ("at most 3", Op.LTE),
            ("over 3", Op.GT),
            ("under 3", Op.LT),
        ]:
            sq = router._structured_query(f"which entries have version major {phrase}", None)
            assert sq is not None, phrase
            assert sq.where == Compare("version_major", op, 3), phrase

    def test_longest_field_name_wins(self, router: RulesRouter) -> None:
        """'version major' contains 'version'. Matching both conjoins a
        predicate on each, and a unit with only version_major matches neither."""
        sq = router._structured_query("which entries have version major greater than 3", None)
        assert sq is not None
        assert sq.where == Compare("version_major", Op.GT, 3)

    def test_categorical_value_becomes_an_equality(self, router: RulesRouter) -> None:
        sq = router._structured_query("which entries have package flask", None)
        assert sq is not None
        assert sq.where == TextMatch("package", "flask", mode="exact")

    def test_bare_field_mention_becomes_existence(self, router: RulesRouter) -> None:
        sq = router._structured_query("how many distinct package values are recorded", None)
        assert sq is not None
        assert sq.where == Exists("package")

    def test_no_field_named_means_no_predicate(self, router: RulesRouter) -> None:
        """Better to fall back to retrieval than to invent a constraint."""
        assert router._structured_query("how do I configure a timeout", None) is None


class TestPredicatesActuallyMatch:
    """The point of a predicate is to select rows. These run the extracted
    predicates against real values, because a predicate that compiles and
    matches nothing is the failure mode that looks like a retrieval problem."""

    @pytest.mark.parametrize(
        ("question", "expected_packages"),
        [
            ("which entries have version major greater than 3", {"click", "attrs"}),
            ("which entries have version major at least 8", {"click", "attrs"}),
            ("which entries have release date before 2024-01-01", {"click"}),
            ("which entries have release date after 2024-01-01", {"flask", "attrs"}),
            ("which entries have package flask", {"flask"}),
            ("how many distinct package values are recorded", {"flask", "click", "attrs"}),
        ],
    )
    def test_extracted_predicate_selects_the_right_rows(
        self, router: RulesRouter, question: str, expected_packages: set[str]
    ) -> None:
        sq = router._structured_query(question, None)
        assert sq is not None and sq.where is not None, question
        got = {r["package"] for r in ROWS if evaluate(sq.where, r)}
        assert got == expected_packages, f"{question}: {sq.where}"


class TestRouting:
    def test_structured_questions_take_the_structured_path(self, router: RulesRouter) -> None:
        for q in (
            "how many entries have a package recorded",
            "which entries have version major greater than 3",
            "which entries have release date before 2023-06-27",
        ):
            d = router.route(Query(text=q), None)  # type: ignore[arg-type]
            assert str(d.path) == RoutePath.STRUCTURED, q
            assert d.structured_query is not None

    def test_prose_questions_take_lookup(self, router: RulesRouter) -> None:
        d = router.route(Query(text="how do I set a read timeout"), None)  # type: ignore[arg-type]
        assert str(d.path) == RoutePath.LOOKUP
        assert d.query_type is QueryType.FACTUAL

    def test_a_structured_question_with_no_structured_index_falls_back_loudly(self) -> None:
        r = RulesRouter(
            {
                "field_lexicon": LEXICON,
                "enable_structured": False,
                "paths": {
                    "structured": {"targets": []},
                    "lookup": {"targets": ["lexical"]},
                    "iterative": {"targets": ["lexical"], "step_budget": 3},
                },
            }
        )
        d = r.route(Query(text="how many entries have a package recorded"), None)  # type: ignore[arg-type]
        assert str(d.path) == RoutePath.LOOKUP
        # The decision log must show the structured path was wanted, not that
        # the question looked like prose.
        assert "structured" in d.reason.lower()


class TestDeclaredFieldTypes:
    """The extraction config states each field's type. Without it the router
    guesses from how a value is written, and every corpus with a patch release
    breaks: "version 1.0.0" reads as the float 1.0, which queries a numeric
    column the value was never written to, so the question matches nothing.
    """

    @pytest.fixture
    def typed(self) -> RulesRouter:
        return RulesRouter(
            {
                "field_lexicon": LEXICON,
                "field_types": {
                    "package": "str",
                    "version": "str",
                    "version_major": "int",
                    "release_date": "date",
                },
                "paths": {
                    "structured": {"targets": ["fields"]},
                    "lookup": {"targets": ["lexical"]},
                    "iterative": {"targets": ["lexical"], "step_budget": 3},
                },
            }
        )

    @pytest.mark.parametrize("value", ["1.0.0", "2.1", "0.9.13", "2024.1.2"])
    def test_a_textual_field_takes_a_textual_value_however_numeric_it_looks(
        self, typed: RulesRouter, value: str
    ) -> None:
        sq = typed._structured_query(f"which entries have version {value}", None)
        assert sq is not None
        assert sq.where == TextMatch("version", value, mode="exact")

    def test_a_numeric_field_still_takes_a_number(self, typed: RulesRouter) -> None:
        sq = typed._structured_query("which entries have version major greater than 3", None)
        assert sq is not None
        assert sq.where == Compare("version_major", Op.GT, 3)

    def test_a_date_field_still_takes_a_date(self, typed: RulesRouter) -> None:
        sq = typed._structured_query("which entries have release date before 2023-06-27", None)
        assert sq is not None
        assert sq.where == Compare("release_date", Op.LT, date(2023, 6, 27))

    def test_typed_predicates_select_the_right_rows(self, typed: RulesRouter) -> None:
        rows = [
            {"package": "flask", "version": "3.1.0", "version_major": 3},
            {"package": "click", "version": "8.1.7", "version_major": 8},
        ]
        sq = typed._structured_query("which entries have version 3.1.0", None)
        assert sq is not None and sq.where is not None
        got = {r["package"] for r in rows if evaluate(sq.where, r)}
        assert got == {"flask"}, sq.where

    def test_the_type_map_comes_from_the_extraction_config(self) -> None:
        """One derivation, so the router's view cannot drift from the schema."""
        from indexer.config import load

        cfg, _ = load("configs/pypi-docs.yaml")
        types = cfg.extracted_field_types()
        assert types["version"] == "str"
        assert types["version_major"] == "int"
        assert types["release_date"] == "date"
        assert set(types) == set(cfg.extracted_field_names())

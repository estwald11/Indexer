"""The rules router on Italian questions.

Before: "fatture con importo superiore a 1000 euro" matched no rule, was
classified as prose and sent to vector search -- while the same question with
"greater than" took the structured path. Every case here is a question an
Italian user types to an agent over a company archive.
"""

from __future__ import annotations

from datetime import date

import pytest

from indexer.core.predicate import AggregationOp, evaluate
from indexer.core.query import Query, QueryType, RoutePath
from indexer.impls.route import RulesRouter

ROWS = [
    {"doc": "f1", "tipo_documento": "fattura", "importo": 1500.0, "fornitore": "Rossi Srl",
     "data_fattura": date(2024, 3, 10)},
    {"doc": "f2", "tipo_documento": "fattura", "importo": 250.0, "fornitore": "Rossi Srl",
     "data_fattura": date(2025, 7, 2)},
    {"doc": "f3", "tipo_documento": "fattura", "importo": 9000.0, "fornitore": "Bianchi Spa",
     "data_fattura": date(2025, 6, 15)},
    {"doc": "c1", "tipo_documento": "contratto", "importo": 50000.0, "fornitore": "Bianchi Spa",
     "data_fattura": date(2024, 1, 20), "data_scadenza": date(2025, 12, 1)},
    {"doc": "c2", "tipo_documento": "contratto", "fornitore": "Verdi Snc",
     "data_scadenza": date(2026, 6, 30)},
]  # fmt: skip


@pytest.fixture
def router() -> RulesRouter:
    return RulesRouter(
        {
            "field_lexicon": [
                "importo",
                "data_fattura",
                "fornitore",
                "tipo_documento",
                "data_scadenza",
            ],
            "field_types": {
                "importo": "float",
                "data_fattura": "date",
                "fornitore": "str",
                "tipo_documento": "str",
                "data_scadenza": "date",
            },
            "field_aliases": {"data_fattura": ["data"], "data_scadenza": ["scadenza"]},
            "value_aliases": {
                "tipo_documento": {"fattura": ["fatture"], "contratto": ["contratti"]}
            },
            "default_date_field": "data_fattura",
            "default_measure": "importo",
            "locale": "it",
            "level": "document",
            "paths": {
                "structured": {"targets": ["fields"]},
                "lookup": {"targets": ["lexical", "dense"]},
                "iterative": {"targets": ["lexical", "dense"], "step_budget": 3},
            },
        }
    )


def _docs(router: RulesRouter, question: str) -> set[str]:
    d = router.route(Query(text=question), None)  # type: ignore[arg-type]
    assert str(d.path) == RoutePath.STRUCTURED, f"{question}: {d.reason}"
    assert d.structured_query is not None and d.structured_query.where is not None
    return {r["doc"] for r in ROWS if evaluate(d.structured_query.where, r)}


class TestItalianQuestionsTakeTheStructuredPath:
    @pytest.mark.parametrize(
        ("question", "docs"),
        [
            ("fatture con importo superiore a 1000 euro", {"f1", "f3"}),
            ("fatture con importo inferiore a 1.000", {"f2"}),
            ("fatture con importo di almeno 1.500,00 €", {"f1", "f3"}),
            ("fatture con data successiva al 30/06/2025", {"f2"}),
            ("fatture con data prima del 1 gennaio 2025", {"f1"}),
            ("fatture emesse nel 2024", {"f1"}),
            ("fatture di giugno 2025", {"f3"}),
            ("fatture sopra i 1.000 euro", {"f1", "f3"}),
            ("contratti in scadenza entro il 31/12/2025", {"c1"}),
            ("elenca le fatture del fornitore 'Bianchi Spa'", {"f3"}),
        ],
    )
    def test_predicate_selects_the_right_documents(
        self, router: RulesRouter, question: str, docs: set[str]
    ) -> None:
        assert _docs(router, question) == docs

    def test_the_same_question_in_english_still_works(self, router: RulesRouter) -> None:
        assert _docs(router, "invoices with importo greater than 1000") == {"f1", "f3", "c1"}

    def test_plural_field_names_are_recognised(self, router: RulesRouter) -> None:
        d = router.route(Query(text="somma degli importi per fornitore"), None)  # type: ignore[arg-type]
        sq = d.structured_query
        assert sq is not None
        assert sq.group_by == ("fornitore",)
        assert [(a.op, a.field) for a in sq.aggregations] == [(AggregationOp.SUM, "importo")]


class TestAggregates:
    def test_quante_counts_documents(self, router: RulesRouter) -> None:
        d = router.route(Query(text="quante fatture sono state emesse nel 2025?"), None)  # type: ignore[arg-type]
        sq = d.structured_query
        assert sq is not None and sq.where is not None
        assert [a.op for a in sq.aggregations] == [AggregationOp.COUNT]
        assert sq.level == "document"
        assert {r["doc"] for r in ROWS if evaluate(sq.where, r)} == {"f2", "f3"}

    def test_average_amount(self, router: RulesRouter) -> None:
        d = router.route(Query(text="qual è l'importo medio delle fatture del 2025"), None)  # type: ignore[arg-type]
        sq = d.structured_query
        assert sq is not None
        assert [(a.op, a.field) for a in sq.aggregations] == [(AggregationOp.AVG, "importo")]

    def test_distinct_suppliers(self, router: RulesRouter) -> None:
        d = router.route(Query(text="quanti fornitori diversi ci sono"), None)  # type: ignore[arg-type]
        sq = d.structured_query
        assert sq is not None
        agg = sq.aggregations[0]
        assert (agg.op, agg.field, agg.distinct) == (AggregationOp.COUNT, "fornitore", True)

    def test_a_field_named_like_an_aggregate_is_not_summed(self) -> None:
        # "importo totale" is a field; the word "totale" in it is not a request
        # to sum anything.
        r = RulesRouter(
            {
                "field_lexicon": ["importo_totale"],
                "field_types": {"importo_totale": "float"},
                "locale": "it",
                "paths": {"structured": {"targets": ["fields"]}},
            }
        )
        sq = r._structured_query("fatture con importo totale superiore a 500", None)
        assert sq is not None
        assert sq.aggregations == ()


class TestProseStaysProse:
    @pytest.mark.parametrize(
        "question",
        [
            "come si configura il timeout di lettura",
            "chi è il referente commerciale di Rossi",
            "cosa prevede la clausola di recesso",
        ],
    )
    def test_questions_without_structure_take_lookup(
        self, router: RulesRouter, question: str
    ) -> None:
        d = router.route(Query(text=question), None)  # type: ignore[arg-type]
        assert str(d.path) == RoutePath.LOOKUP, d.reason
        assert d.query_type is QueryType.FACTUAL

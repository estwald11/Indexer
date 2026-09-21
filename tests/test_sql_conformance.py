"""The SQL compiler must agree with the reference evaluator.

``indexer.core.predicate.evaluate`` is the AST's executable definition;
``compile_predicate`` turns the same AST into SQL. If the two disagree, the same
question returns different answers depending on which store answered it -- and
the disagreement would surface as "the numbers are sometimes wrong", months
later, with nothing to attribute it to.

So the test is differential: build units with known fields, run every predicate
both ways, require identical unit sets.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.ids import DocumentId, hash_text, make_unit_id
from indexer.core.predicate import (
    And,
    Compare,
    Exists,
    In,
    Not,
    Op,
    Or,
    Predicate,
    StructuredQuery,
    TextMatch,
    evaluate,
)
from indexer.core.provenance import Provenance, Span
from indexer.core.stages import IndexQuery, StageContext
from indexer.core.unit import EnrichedUnit, Enrichment, Unit
from indexer.impls.index_structured import SqliteStructuredIndex

ROWS: list[dict] = [
    {"package": "requests", "version_major": 2, "released": date(2024, 5, 1), "stable": True},
    {"package": "urllib3", "version_major": 2, "released": date(2023, 1, 15), "stable": True},
    {"package": "flask", "version_major": 3, "released": date(2025, 11, 2), "stable": False},
    {"package": "click", "version_major": 8, "released": date(2022, 7, 30), "stable": True},
    {"package": "attrs", "version_major": 25},  # released/stable absent entirely
]

PREDICATES: list[Predicate] = [
    Compare("version_major", Op.EQ, 2),
    Compare("version_major", Op.NE, 2),
    Compare("version_major", Op.GT, 2),
    Compare("version_major", Op.GTE, 3),
    Compare("version_major", Op.LT, 8),
    Compare("version_major", Op.LTE, 2),
    Compare("released", Op.LT, date(2024, 1, 1)),
    Compare("released", Op.GTE, date(2024, 1, 1)),
    Compare("package", Op.EQ, "flask"),
    Compare("stable", Op.EQ, True),
    Compare("stable", Op.EQ, False),
    In("package", ("requests", "click")),
    In("version_major", (2, 3)),
    Exists("released"),
    Exists("released", present=False),
    Exists("stable", present=False),
    TextMatch("package", "ur"),
    TextMatch("package", "re", mode="prefix"),
    TextMatch("package", "flask", mode="exact"),
    And((Compare("version_major", Op.GTE, 2), Exists("released"))),
    Or((Compare("package", Op.EQ, "attrs"), Compare("version_major", Op.GT, 7))),
    Not(Compare("version_major", Op.EQ, 2)),
    And(
        (
            Or((TextMatch("package", "re"), TextMatch("package", "fl"))),
            Not(Compare("released", Op.LT, date(2024, 1, 1))),
        )
    ),
]


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> SqliteStructuredIndex:
    path: Path = tmp_path_factory.mktemp("sql") / "f.db"
    idx = SqliteStructuredIndex({"path": str(path)}, name="fields")
    units = []
    for i, fields in enumerate(ROWS):
        text = f"unit {i} for {fields['package']}"
        did = DocumentId(f"d{i}")
        u = Unit(
            unit_id=make_unit_id(did, hash_text(text)),
            document_id=did,
            text=text,
            provenance=Provenance(document_id=did, span=Span(0, len(text))),
        )
        units.append(
            EnrichedUnit(
                unit=u,
                enrichments={"f": Enrichment(enricher="f", fingerprint="1", fields=dict(fields))},
            )
        )
    ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
    idx.upsert(units, ctx)
    idx.flush()
    idx._units = units  # type: ignore[attr-defined]  -- for the reference side
    return idx


def _reference(pred: Predicate) -> set[str]:
    """What the in-memory definition says the answer is."""
    out = set()
    for i, fields in enumerate(ROWS):
        try:
            if evaluate(pred, dict(fields)):
                out.add(f"row{i}")
        except TypeError:
            pytest.fail(f"reference evaluator raised on {pred}")
    return out


def _via_sql(index: SqliteStructuredIndex, pred: Predicate) -> set[str]:
    ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
    rs = index.structured_query(StructuredQuery(where=pred, select=("package",)), ctx)
    names = {r[0] for r in rs.rows if r[0] is not None}
    return {f"row{i}" for i, f in enumerate(ROWS) if f["package"] in names}


@pytest.mark.parametrize("pred", PREDICATES, ids=lambda p: type(p).__name__ + str(hash(str(p)))[:4])
def test_sql_agrees_with_reference_evaluator(index: SqliteStructuredIndex, pred: Predicate) -> None:
    assert _via_sql(index, pred) == _reference(pred), (
        f"SQL and the reference evaluator disagree on {pred}. The same question "
        f"would get different answers from different stores."
    )


class TestStructuredQueryShape:
    def test_rows_cite_their_sources(self, index: SqliteStructuredIndex) -> None:
        """A number in an answer must be as citable as a passage."""
        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        rs = index.structured_query(
            StructuredQuery(where=Exists("released"), select=("package", "released")), ctx
        )
        assert rs.columns == ("package", "released")
        assert len(rs.rows) == 4
        assert all(src for src in rs.sources)

    def test_aggregation(self, index: SqliteStructuredIndex) -> None:
        from indexer.core.predicate import Aggregation, AggregationOp

        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        rs = index.structured_query(
            StructuredQuery(
                where=Exists("version_major"),
                aggregations=(Aggregation(AggregationOp.MAX, "version_major"),),
            ),
            ctx,
        )
        assert rs.rows[0][0] == 25

    def test_filters_push_down_to_search(self, index: SqliteStructuredIndex) -> None:
        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        rl = index.search(
            IndexQuery(text="", top_k=10, filters=Compare("version_major", Op.GT, 7)), ctx
        )
        assert len(rl.hits) == 2  # click (8) and attrs (25)


class TestInjection:
    def test_values_are_bound_not_interpolated(self, index: SqliteStructuredIndex) -> None:
        """A structured route receives user text; values must never be inlined."""
        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        nasty = "flask'; DROP TABLE units; --"
        rs = index.structured_query(
            StructuredQuery(where=Compare("package", Op.EQ, nasty), select=("package",)), ctx
        )
        assert rs.rows == ()
        # the table is still there
        assert index.stats().unit_count == len(ROWS)


class TestUnfilteredSearchIsEmpty:
    """A structured index has no text ranking, so an unfiltered text search must
    return nothing rather than arbitrary rows.

    The rows it used to return were the first k by unit id -- identical for
    every query, each carrying rank-1 weight into fusion. That is not a weak
    signal, it is a constant one, and it displaces real top hits the same way
    for every query in a set. It surfaced in the no-router arm, where routing is
    off and every index is queried for everything.
    """

    def test_unfiltered_text_search_returns_nothing(self, index: SqliteStructuredIndex) -> None:
        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        rl = index.search(IndexQuery(text="flask sessions cookies", top_k=10), ctx)
        assert rl.hits == ()

    def test_filtered_search_still_returns_matches(self, index: SqliteStructuredIndex) -> None:
        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        rl = index.search(
            IndexQuery(text="", top_k=10, filters=Compare("version_major", Op.GT, 7)), ctx
        )
        assert len(rl.hits) == 2

    def test_unit_id_restriction_is_honoured(self, index: SqliteStructuredIndex) -> None:
        """The iterative path and rerank candidate sets narrow by unit id."""
        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        everything = index.structured_query(
            StructuredQuery(where=Exists("package"), select=("package",)), ctx
        )
        assert everything.rows
        some = index.search(
            IndexQuery(text="", top_k=10, unit_ids=[u for u in index.all_unit_ids()[:2]]), ctx
        )
        assert len(some.hits) == 2

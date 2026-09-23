"""Document-level structured queries, schema description, and metadata as fields.

"How many invoices over 1,000 euros from supplier X" is a question about
documents. Answered per unit it counts chunks, and it fails outright when the
supplier is named in one chunk and the total in another. These tests pin the
document level, and the two things an agent needs to use it: the schema, and
scanner facts (tenant, ACL) that reach the index without an enricher copying
them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.ids import DocumentId, hash_text, make_unit_id
from indexer.core.predicate import (
    Aggregation,
    AggregationOp,
    And,
    Compare,
    In,
    Op,
    StructuredQuery,
    TextMatch,
)
from indexer.core.provenance import Provenance, Span
from indexer.core.stages import IndexQuery, StageContext
from indexer.core.unit import EnrichedUnit, Enrichment, Unit
from indexer.impls.index_structured import SqliteStructuredIndex

CTX = StageContext(cache=NullCache(), accountant=InMemoryAccountant())


def _unit(doc: str, ordinal: int, text: str, fields: dict, metadata: dict | None = None):  # type: ignore[no-untyped-def,type-arg]
    did = DocumentId(doc)
    return EnrichedUnit(
        unit=Unit(
            unit_id=make_unit_id(did, hash_text(text)),
            document_id=did,
            text=text,
            provenance=Provenance(document_id=did, span=Span(0, len(text))),
            ordinal=ordinal,
            metadata=metadata or {},
        ),
        enrichments={"x": Enrichment(enricher="x", fingerprint="1", fields=fields)},
    )


@pytest.fixture
def index(tmp_path: Path) -> SqliteStructuredIndex:
    idx = SqliteStructuredIndex({"path": str(tmp_path / "f.db")})
    acme = {"tenant": "acme", "acl": ["group:finance"]}
    idx.upsert(
        [
            # Invoice 1: supplier on the first page, total on the last.
            _unit("inv-1", 0, "Fornitore Rossi Srl", {"fornitore": "Rossi Srl"}, acme),
            _unit("inv-1", 1, "Totale 1.500,00", {"importo": 1500.0}, acme),
            # Invoice 2: same supplier, small amount, mentioned twice.
            _unit(
                "inv-2",
                0,
                "Fornitore Rossi Srl, importo 200",
                {"fornitore": "Rossi Srl", "importo": 200},
                acme,
            ),
            _unit("inv-2", 1, "Riepilogo: importo 200", {"importo": 200}, acme),
            # Invoice 3: another supplier, another tenant.
            _unit(
                "inv-3",
                0,
                "Fornitore Bianchi Spa totale 9000",
                {"fornitore": "Bianchi Spa", "importo": 9000, "data": date(2025, 3, 1)},
                {"tenant": "globex", "acl": ["group:legal", "user:anna"]},
            ),
        ],
        CTX,
    )
    idx.flush()
    return idx


class TestDocumentLevel:
    def test_a_predicate_spanning_units_matches_the_document(
        self, index: SqliteStructuredIndex
    ) -> None:
        pred = And((TextMatch("fornitore", "rossi"), Compare("importo", Op.GT, 1000)))
        per_unit = index.structured_query(StructuredQuery(where=pred), CTX)
        per_doc = index.structured_query(StructuredQuery(where=pred, level="document"), CTX)
        # No single unit names both, so the unit level finds nothing...
        assert per_unit.is_empty()
        # ...while the document does both, on different pages.
        assert [r[0] for r in per_doc.rows] == ["inv-1"]
        assert per_doc.sources and per_doc.sources[0]

    def test_counting_counts_documents_not_chunks(self, index: SqliteStructuredIndex) -> None:
        count = (Aggregation(AggregationOp.COUNT),)
        by_rossi = TextMatch("fornitore", "rossi")
        units = index.structured_query(StructuredQuery(where=by_rossi, aggregations=count), CTX)
        docs = index.structured_query(
            StructuredQuery(where=by_rossi, aggregations=count, level="document"), CTX
        )
        assert docs.rows == ((2,),)
        assert units.rows == ((2,),)  # the two units naming Rossi, by coincidence equal
        amounts = index.structured_query(
            StructuredQuery(
                where=Compare("importo", Op.GTE, 200), aggregations=count, level="document"
            ),
            CTX,
        )
        # inv-2 mentions its amount in two units: one document all the same.
        assert amounts.rows == ((3,),)

    def test_group_by_at_document_level(self, index: SqliteStructuredIndex) -> None:
        rs = index.structured_query(
            StructuredQuery(
                select=("fornitore",),
                group_by=("fornitore",),
                aggregations=(Aggregation(AggregationOp.COUNT),),
                order_by=(("fornitore", False),),
                level="document",
            ),
            CTX,
        )
        assert rs.as_dicts() == [
            {"fornitore": "Bianchi Spa", "count": 1},
            {"fornitore": "Rossi Srl", "count": 2},
        ]

    def test_deleting_a_units_refreshes_its_document(self, index: SqliteStructuredIndex) -> None:
        inv1 = [u for u in index.all_unit_ids()]
        before = index.document_fields(["inv-1"])["inv-1"]
        assert before["importo"] == 1500.0
        # Remove every unit of inv-1: the document disappears with them.
        rows = index._conn.execute("SELECT unit_id FROM units WHERE document_id='inv-1'")
        index.delete([r["unit_id"] for r in rows], CTX)
        assert "inv-1" not in index.document_fields(["inv-1"])
        assert len(index.all_unit_ids()) == len(inv1) - 2


class TestIntrospection:
    def test_describe_schema_reports_types_ranges_and_examples(
        self, index: SqliteStructuredIndex
    ) -> None:
        schema = {f["name"]: f for f in index.describe_schema()}
        assert set(schema) >= {"fornitore", "importo", "data", "tenant", "acl"}
        assert set(schema["importo"]["types"]) == {"float", "int"}
        assert (schema["importo"]["min"], schema["importo"]["max"]) == (200, 9000)
        assert schema["importo"]["documents"] == 3
        assert schema["data"]["types"] == ["date"]
        assert schema["fornitore"]["examples"][0] == "Rossi Srl"

    def test_document_fields_are_typed_and_multi_valued(self, index: SqliteStructuredIndex) -> None:
        cards = index.document_fields(["inv-3", "inv-2"])
        assert cards["inv-3"]["data"] == date(2025, 3, 1)
        assert cards["inv-3"]["acl"] == ("group:legal", "user:anna")
        assert cards["inv-2"]["importo"] == 200


class TestMetadataIsFilterable:
    def test_scanner_metadata_filters_without_an_enricher_copying_it(
        self, index: SqliteStructuredIndex
    ) -> None:
        rl = index.search(IndexQuery(text="", filters=Compare("tenant", Op.EQ, "globex")), CTX)
        assert {h.document_id for h in rl.hits} == {"inv-3"}
        rl = index.search(IndexQuery(text="", filters=In("acl", ("group:finance",))), CTX)
        assert {h.document_id for h in rl.hits} == {"inv-1", "inv-2"}

    def test_metadata_wins_over_an_extracted_field_of_the_same_name(self) -> None:
        # A document cannot move itself to another tenant by containing text an
        # extractor reads as a tenant.
        eu = _unit("d", 0, "tenant: globex", {"tenant": "globex"}, {"tenant": "acme"})
        assert eu.filter_fields()["tenant"] == "acme"

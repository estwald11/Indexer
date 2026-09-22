"""Tenant isolation: a caller's scope must hold on every path and in every cache.

The frame's multi-tenant story is a caller filter (``Query.filters``) plus
scanner metadata. Both are only as good as the weakest path that ignores them,
and an archive shared by several companies -- or several departments of one --
is exactly where a leak is a incident rather than a quality problem. Each test
here is a reproduction of a leak that existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indexer.core.predicate import Compare, Op
from indexer.core.query import RoutePath
from indexer.pipeline import assemble

CONFIG = """
schema_version: 1
project: {{name: tenancy}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params:
        root: {data}
        include: ["**/*.md"]
        path_metadata: {{tenant: '^([^/]+)/'}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural, max_tokens: 200, params: {{merge_below_tokens: 0}}}}
  enrich:
    enabled: true
    enrichers:
      - impl: regex_fields
        scope: unit
        params:
          from_metadata: [tenant]
          fields:
            amount: {{pattern: 'amount ([0-9]+)', type: int}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25_memory}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
"""


def _workspace(tmp_path: Path, files: dict[str, str]) -> Path:
    data = tmp_path / "data"
    for rel, body in files.items():
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG.format(root=tmp_path.as_posix(), data=data.as_posix()))
    return cfg


ACME = Compare("tenant", Op.EQ, "acme")
GLOBEX = Compare("tenant", Op.EQ, "globex")


class TestStructuredPathHonoursCallerScope:
    @pytest.fixture
    def assembly(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        cfg = _workspace(
            tmp_path,
            {
                "acme/invoice.md": "# Invoice\n\nThe invoice amount 1500 is due in March.\n",
                "globex/invoice.md": "# Invoice\n\nThe invoice amount 9000 is due in April.\n",
            },
        )
        a = assemble(cfg)
        assert a.ingestion().build().ok
        return a

    def test_records_from_other_tenants_are_not_returned(self, assembly) -> None:  # type: ignore[no-untyped-def]
        resp = assembly.query_engine().query("invoices with amount greater than 1000", filters=ACME)
        assert str(resp.decision.path) == RoutePath.STRUCTURED
        assert resp.records is not None
        assert [r["amount"] for r in resp.records.as_dicts()] == [1500]

    def test_unscoped_query_still_sees_everything(self, assembly) -> None:  # type: ignore[no-untyped-def]
        resp = assembly.query_engine().query("invoices with amount greater than 1000")
        assert resp.records is not None
        assert sorted(r["amount"] for r in resp.records.as_dicts()) == [1500, 9000]


def _lexical_docs(a, text: str, filters=None) -> list[str]:  # type: ignore[no-untyped-def]
    from indexer.core.accounting import InMemoryAccountant
    from indexer.core.cache import NullCache
    from indexer.core.stages import IndexQuery, StageContext

    ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
    rl = a.indexes["lexical"].search(IndexQuery(text=text, top_k=20, filters=filters), ctx)
    return sorted(h.provenance.source_uri.rsplit("/data/", 1)[-1] for h in rl.hits)


class TestCachesDoNotCollapseDocuments:
    """Content-addressed caches must share work, never identity."""

    def test_identical_files_in_two_tenants_are_two_documents(self, tmp_path: Path) -> None:
        body = "# Terms\n\nPayment is due within 30 days of the invoice date.\n"
        cfg = _workspace(tmp_path, {"acme/terms.md": body, "globex/terms.md": body})
        a = assemble(cfg)
        res = a.ingestion().build()
        assert res.ok
        # Two documents in, two documents indexed. The shared parse used to
        # carry the first document's id, so the second never reached an index.
        assert a.indexes["lexical"].stats().unit_count == 2
        assert _lexical_docs(a, "payment due", GLOBEX) == ["globex/terms.md"]
        assert _lexical_docs(a, "payment due", ACME) == ["acme/terms.md"]

    def test_shared_paragraph_keeps_each_tenants_metadata(self, tmp_path: Path) -> None:
        shared = "Personal data is processed according to the GDPR privacy notice."
        privacy = f"## Privacy\n\n{shared}\n"
        cfg = _workspace(
            tmp_path,
            {
                "acme/contract.md": f"# Contract\n\nAcme supplies widgets.\n\n{privacy}",
                "globex/offer.md": f"# Offer\n\nGlobex sells gadgets.\n\n{privacy}",
            },
        )
        a = assemble(cfg)
        assert a.ingestion().build().ok
        for uid in a.unit_store.all_ids():
            eu = a.unit_store.get(uid)
            assert eu is not None
            # Before the fix the globex unit's tenant came back from the cache
            # as "acme", and it answered acme's filtered queries.
            assert eu.fields().get("tenant") == eu.unit.metadata["tenant"]
        assert _lexical_docs(a, "privacy GDPR", ACME) == ["acme/contract.md"]
        assert _lexical_docs(a, "privacy GDPR", GLOBEX) == ["globex/offer.md"]

    def test_same_bytes_routed_to_different_parsers_are_parsed_by_each(
        self, tmp_path: Path
    ) -> None:
        cfg_text = CONFIG.replace(
            'include: ["**/*.md"]', 'include: ["**/*.md", "**/*.txt"]'
        ).replace(
            "parse: {{enabled: true, default: {{impl: markdown}}}}",
            "parse: {{enabled: true, routes: [{{when: {{media_type: text/markdown}}, "
            "impl: markdown}}], default: {{impl: text}}}}",
        )
        data = tmp_path / "data" / "acme"
        data.mkdir(parents=True)
        body = "# Heading\n\nSome text under the heading.\n"
        (data / "x.md").write_text(body)
        (data / "y.txt").write_text(body)
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            cfg_text.format(root=tmp_path.as_posix(), data=(tmp_path / "data").as_posix())
        )
        a = assemble(cfg)
        assert a.ingestion().build().ok
        by_name = {}
        for uid in a.unit_store.all_ids():
            eu = a.unit_store.get(uid)
            assert eu is not None
            by_name.setdefault(eu.unit.metadata["name"], []).append(eu.unit)
        assert set(by_name) == {"x.md", "y.txt"}
        # Markdown recovers the heading; the text parser does not.
        assert by_name["x.md"][0].section_path == ("Heading",)
        assert by_name["y.txt"][0].section_path == ()

    def test_the_shared_parse_entry_carries_no_identity(self, tmp_path: Path) -> None:
        body = "# Terms\n\nPayment is due within 30 days.\n"
        cfg = _workspace(tmp_path, {"acme/terms.md": body})
        a = assemble(cfg)
        assert a.ingestion().build().ok
        from indexer.pipeline.ingest import parse_cache_key

        doc = next(iter(a.scanner().scan()))
        raw = a.cache.get(parse_cache_key(a.parser(), doc))
        assert raw is not None
        # Neither the tenant nor the location of the copy that happened to be
        # parsed first may live in an entry every copy will be served from.
        assert b"acme" not in raw
        assert doc.document_id.encode() not in raw


class TestRebinding:
    def test_parser_metadata_cannot_override_scanner_facts(self) -> None:
        from indexer.core.document import SourceDocument
        from indexer.core.ids import DocumentId, hash_text
        from indexer.impls.parse import _Builder
        from indexer.pipeline.ingest import rebind_parsed

        b = _Builder("", "")
        b.add("hello", "paragraph")
        parsed = b.finish(hash_text("hello"))
        doc = SourceDocument(
            document_id=DocumentId("doc-1"),
            source_uri="file:///archive/acme/x.eml",
            content_hash=hash_text("hello"),
            media_type="message/rfc822",
            size_bytes=5,
            metadata={"tenant": "acme"},
        )
        # A document whose headers claim another tenant must not move there.
        out = rebind_parsed(parsed, doc, {"tenant": "globex", "subject": "hi"})
        assert out.metadata == {"tenant": "acme", "subject": "hi"}
        assert out.document_id == "doc-1"
        assert all(blk.provenance.document_id == "doc-1" for blk in out.blocks)
        assert all(blk.provenance.source_uri == doc.source_uri for blk in out.blocks)

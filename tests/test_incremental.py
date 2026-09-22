"""Incremental rebuilds must leave every store exactly as a clean build would.

"Adding 10 documents to 500 reprocesses 10" is only a feature if the other 490
are still *right* afterwards. Each test edits the corpus or the config, rebuilds
incrementally, and checks a property a clean build has by construction -- the
failures here were all found by doing that on a real edit and looking.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.predicate import Compare, Op
from indexer.core.stages import IndexQuery, StageContext
from indexer.pipeline import assemble

CONFIG = """
schema_version: 1
project: {{name: incremental}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params: {{root: {data}, include: ["**/*.md"]}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural, max_tokens: 200, params: {{merge_below_tokens: 0}}}}
  enrich:
    enabled: true
    enrichers:
      - {{impl: section_prefix, scope: unit}}
      - impl: regex_fields
        scope: unit
        params:
          fields:
            amount: {{pattern: '{pattern}', type: int}}
            paid_on: {{pattern: '(\\d{{4}}-\\d{{2}}-\\d{{2}})', type: date}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25_memory}}
      - {{name: dense, kind: dense, impl: hash_embedding, params: {{dim: {dim}}}}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical, dense], step_budget: 1}}
      iterative: {{targets: [lexical, dense], step_budget: 3}}
"""

DOC = (
    "# Ledger\n\n## Payments\n\nThe amount 100 was paid on 2024-01-05.\n\n"
    "## Notes\n\nAnother paragraph that stays the same forever.\n"
)
CTX = StageContext(cache=NullCache(), accountant=InMemoryAccountant())


def _assemble(root: Path, *, pattern: str = "amount ([0-9]+)", dim: int = 64):  # type: ignore[no-untyped-def]
    cfg = root / "c.yaml"
    cfg.write_text(
        CONFIG.format(
            root=root.as_posix(), data=(root / "data").as_posix(), pattern=pattern, dim=dim
        )
    )
    return assemble(cfg)


def _setup(tmp_path: Path, body: str = DOC) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ledger.md").write_text(body)
    return tmp_path


def _stored_units(a):  # type: ignore[no-untyped-def]
    return [eu for uid in a.unit_store.all_ids() if (eu := a.unit_store.get(uid)) is not None]


def _parsed(a):  # type: ignore[no-untyped-def]
    doc = next(iter(a.scanner().scan()))
    return a.parser().parse(doc, CTX)


class TestEditsAboveAUnit:
    def test_spans_follow_the_text_after_an_insertion(self, tmp_path: Path) -> None:
        root = _setup(tmp_path)
        assert _assemble(root).ingestion().build().ok
        (root / "data" / "ledger.md").write_text(
            DOC.replace("# Ledger\n\n", "# Ledger\n\nA new introduction, inserted on top.\n\n")
        )
        a = _assemble(root)
        assert a.ingestion().build().ok

        parsed = _parsed(a)
        for eu in _stored_units(a):
            span = eu.unit.provenance.span
            # Unchanged text keeps its id -- and used to keep its old span, so
            # every citation from it pointed at the wrong offset.
            assert parsed.text[span.start : span.end] == eu.unit.text

        for name in ("lexical", "dense"):
            rl = a.indexes[name].search(IndexQuery(text="paragraph stays forever", top_k=5), CTX)
            top = rl.hits[0]
            assert parsed.text[top.provenance.span.start : top.provenance.span.end] == (
                "Another paragraph that stays the same forever."
            ), name

    def test_a_renamed_heading_reaches_section_path_and_context(self, tmp_path: Path) -> None:
        root = _setup(tmp_path)
        assert _assemble(root).ingestion().build().ok
        (root / "data" / "ledger.md").write_text(DOC.replace("## Payments", "## Settlements"))
        a = _assemble(root)
        assert a.ingestion().build().ok

        paid = next(eu for eu in _stored_units(a) if "amount" in eu.unit.text)
        assert paid.unit.section_path == ("Ledger", "Settlements")
        assert paid.enrichments["section_prefix"].context == "Ledger > Settlements"


class TestConfigChanges:
    def test_a_corrected_extraction_rule_reaches_the_structured_index(self, tmp_path: Path) -> None:
        root = _setup(tmp_path)
        assert _assemble(root).ingestion().build().ok
        # Same surface text, different field value: the index used to compare
        # surface hashes, skip the write, and keep amount=100 forever.
        a = _assemble(root, pattern="amount ([0-9]+)0")
        res = a.ingestion().build()
        assert [c.kind.value for c in res.plan] == ["restaged"]

        rows = (
            a.indexes["fields"]
            ._conn.execute("SELECT v_int FROM fields WHERE name = 'amount'")
            .fetchall()
        )
        assert [r[0] for r in rows] == [10]
        rl = a.indexes["lexical"].search(
            IndexQuery(text="amount paid", top_k=5, filters=Compare("amount", Op.EQ, 10)), CTX
        )
        assert len(rl.hits) == 1

    def test_changing_the_embedding_dimension_rebuilds_the_dense_index(
        self, tmp_path: Path
    ) -> None:
        root = _setup(tmp_path)
        assert _assemble(root, dim=64).ingestion().build().ok
        a = _assemble(root, dim=128)
        assert a.ingestion().build().ok

        resp = a.query_engine().query("paragraph stays the same forever")
        dense = next(rl for rl in resp.retrieved if rl.source == "dense")
        # The old 64-dimension vectors used to survive the restage, and every
        # query raised inside the dense index -- degraded to lexical-only with
        # nothing but an "error:ValueError" fingerprint to say so.
        assert not dense.fingerprint.startswith("error:")
        assert dense.hits
        assert all(len(v) == 128 for v in a.indexes["dense"]._vecs.values())

    def test_an_unchanged_config_rebuild_writes_nothing(self, tmp_path: Path) -> None:
        root = _setup(tmp_path)
        assert _assemble(root).ingestion().build().ok
        res = _assemble(root).ingestion().build()
        assert res.manifest.corpus.units_written == 0
        assert res.manifest.corpus.documents_unchanged == 1


class TestTypedFilters:
    def test_date_filters_match_on_every_index_before_and_after_a_reload(
        self, tmp_path: Path
    ) -> None:
        root = _setup(tmp_path)
        a = _assemble(root)
        assert a.ingestion().build().ok
        after_2024 = Compare("paid_on", Op.GTE, date(2024, 1, 1))

        def hits(assembly) -> dict[str, int]:  # type: ignore[no-untyped-def]
            return {
                name: len(
                    assembly.indexes[name]
                    .search(IndexQuery(text="amount paid", top_k=5, filters=after_2024), CTX)
                    .hits
                )
                for name in ("lexical", "dense", "fields")
            }

        # In memory, as built...
        assert hits(a) == {"lexical": 1, "dense": 1, "fields": 1}
        # ...and loaded back from disk, where dates used to become strings.
        assert hits(_assemble(root)) == {"lexical": 1, "dense": 1, "fields": 1}

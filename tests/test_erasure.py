"""Removing a document removes it from the cache too.

The indexes and the unit store always forgot a removed document. The cache did
not: its parse (the full text), its segmentation and every enrichment (LLM
summaries, extracted fields) stayed on disk under content-addressed keys that no
build would ever ask for again. For an archive holding personal data that makes
an erasure request impossible to honour, so the cache now keeps references per
document and deletes what nothing uses any more.
"""

from __future__ import annotations

from pathlib import Path

from indexer.pipeline import assemble

CONFIG = """
schema_version: 1
project: {{name: erasure}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params: {{root: {data}, include: ["**/*.md"]}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural, max_tokens: 200}}
  enrich:
    enabled: true
    enrichers:
      - {{impl: section_prefix, scope: unit}}
      - {{impl: extractive_context, scope: document}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25_memory}}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {{targets: []}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
cache:
  purge_unreferenced: {purge}
"""

KEEP = "# Kept\n\nThis document about Quokka habitats stays in the archive.\n"
GONE = "# Zanzibar\n\nMario Rossi, born in Zanzibar, asked to be forgotten.\n"


def _setup(tmp_path: Path, files: dict[str, str], *, purge: bool = True):  # type: ignore[no-untyped-def]
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for name, body in files.items():
        (data / name).write_text(body)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        CONFIG.format(root=tmp_path.as_posix(), data=data.as_posix(), purge=str(purge).lower())
    )
    return cfg


def _cache_mentions(tmp_path: Path, needle: str) -> int:
    return sum(
        1
        for p in (tmp_path / "cache").rglob("*")
        if p.is_file() and needle.encode() in p.read_bytes()
    )


class TestRemoval:
    def test_a_removed_document_leaves_nothing_in_the_cache(self, tmp_path: Path) -> None:
        cfg = _setup(tmp_path, {"keep.md": KEEP, "gone.md": GONE})
        assert assemble(cfg).ingestion().build().ok
        assert _cache_mentions(tmp_path, "Zanzibar") > 0

        (tmp_path / "data" / "gone.md").unlink()
        res = assemble(cfg).ingestion().build()

        assert res.manifest.corpus.documents_removed == 1
        assert res.manifest.corpus.cache_entries_purged > 0
        assert _cache_mentions(tmp_path, "Zanzibar") == 0
        # And only what the removed document used: the other one is untouched.
        assert _cache_mentions(tmp_path, "Quokka") > 0

    def test_the_previous_version_of_an_edited_document_is_purged(self, tmp_path: Path) -> None:
        cfg = _setup(tmp_path, {"gone.md": GONE})
        assert assemble(cfg).ingestion().build().ok
        (tmp_path / "data" / "gone.md").write_text(GONE.replace("Mario Rossi", "A person"))
        assert assemble(cfg).ingestion().build().ok
        # The redaction reached the cache, not just the index.
        assert _cache_mentions(tmp_path, "Mario Rossi") == 0

    def test_an_entry_still_used_by_another_copy_survives(self, tmp_path: Path) -> None:
        cfg = _setup(tmp_path, {"a.md": GONE, "b.md": GONE})
        assert assemble(cfg).ingestion().build().ok
        (tmp_path / "data" / "a.md").unlink()
        a = assemble(cfg)
        assert a.ingestion().build().ok
        # b.md shares a's parse; it must come from the cache, not be re-parsed.
        doc = next(iter(a.scanner().scan()))
        from indexer.pipeline.ingest import parse_cache_key

        assert a.cache.get(parse_cache_key(a.parser(), doc)) is not None

    def test_purging_can_be_turned_off(self, tmp_path: Path) -> None:
        cfg = _setup(tmp_path, {"keep.md": KEEP, "gone.md": GONE}, purge=False)
        assert assemble(cfg).ingestion().build().ok
        (tmp_path / "data" / "gone.md").unlink()
        res = assemble(cfg).ingestion().build()
        assert res.manifest.corpus.cache_entries_purged == 0
        assert _cache_mentions(tmp_path, "Zanzibar") > 0


class TestSweep:
    def test_sweep_removes_entries_no_document_references(self, tmp_path: Path) -> None:
        cfg = _setup(tmp_path, {"keep.md": KEEP})
        a = assemble(cfg)
        assert a.ingestion().build().ok
        # An entry no build recorded -- as a crashed build would leave behind.
        a.cache.put("sha256:" + "ab" * 32, b"orphaned: Zanzibar")
        assert _cache_mentions(tmp_path, "Zanzibar") == 1

        removed = a.ingestion().sweep_cache()
        assert removed == 1
        assert _cache_mentions(tmp_path, "Zanzibar") == 0
        assert _cache_mentions(tmp_path, "Quokka") > 0
        # Nothing referenced was swept: a rebuild is a pure cache hit.
        res = assemble(cfg).ingestion().build()
        assert res.manifest.corpus.documents_unchanged == 1


class TestLedgerDeletion:
    def test_a_deletion_survives_a_crash_before_compaction(self, tmp_path: Path) -> None:
        from indexer.core.ids import BuildId, ContentHash, DocumentId
        from indexer.core.ledger import DocumentRecord
        from indexer.pipeline.stores import JsonLedger

        path = tmp_path / "ledger.json"
        ledger = JsonLedger(path)
        ledger.begin_build(BuildId("b1"))
        for name in ("a", "b"):
            ledger.put(
                DocumentRecord(
                    document_id=DocumentId(name),
                    source_uri=name,
                    content_hash=ContentHash("h"),
                    unit_ids=(),
                    stage_keys={},
                    build_id=BuildId("b1"),
                )
            )
        ledger.commit_build(BuildId("b1"))

        ledger.begin_build(BuildId("b2"))
        ledger.delete(DocumentId("a"))
        # No commit: the process dies here. The tombstone is in the journal.
        reloaded = JsonLedger(path)
        assert reloaded.document_ids() == frozenset({DocumentId("b")})

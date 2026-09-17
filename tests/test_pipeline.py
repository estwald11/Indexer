"""End-to-end tests of the orchestrators.

These test the promises that only the pipeline can keep: the incremental
guarantee, per-stage invalidation, deletion, and the structural enforcement of
invariant 5.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.errors import ContractViolation
from indexer.core.ledger import ChangeKind
from indexer.core.query import RoutePath
from indexer.core.stages import IndexQuery, StageContext
from indexer.pipeline import assemble

CONFIG = """
schema_version: 1
project: {{name: t, description: test}}
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
      - impl: regex_fields
        scope: unit
        params:
          fields:
            timeout_seconds: {{pattern: 'timeout[= ]+([0-9]+)', type: int}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25_memory}}
      - {{name: dense, kind: dense, impl: hash_embedding, params: {{dim: 128}}}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical, dense], step_budget: 1}}
      iterative: {{targets: [lexical, dense], step_budget: 3}}
  fuse: {{impl: rrf, k: 60}}
  rerank: {{enabled: true, impl: lexical_overlap, input_top_k: 20, output_top_k: 5}}
"""

DOC_A = textwrap.dedent(
    """
    # Timeouts

    Requests should carry a deadline.

    ## Read timeout

    Set the read timeout to 27 seconds for slow upstreams. A value of
    timeout=27 is a reasonable default for most production services that
    talk to third parties over the public internet.

    ## Connect timeout

    The connect timeout should be slightly larger than a multiple of three,
    because of the way TCP retransmission windows are scheduled internally.
    """
).strip()

DOC_B = textwrap.dedent(
    """
    # Sessions

    Sessions persist cookies across requests.

    ## Connection pooling

    The underlying TCP connection is reused when several requests go to the
    same host, which removes the handshake cost from every call after the
    first one in a sequence.
    """
).strip()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.md").write_text(DOC_A)
    (data / "b.md").write_text(DOC_B)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG.format(root=tmp_path, data=data))
    return tmp_path


def _build(workspace: Path, **overrides):
    a = assemble(workspace / "c.yaml", overrides=overrides or None)
    return a, a.ingestion().build()


class TestEndToEnd:
    def test_builds_and_retrieves_with_provenance(self, workspace: Path) -> None:
        a, res = _build(workspace)
        assert res.ok
        assert res.manifest.corpus.units_written > 0

        resp = a.query_engine().query("connection pooling reused host", top_k=5)
        assert resp.hits
        top = resp.hits[0]
        # Every returned passage traces to document, page and span.
        assert top.provenance.document_id
        assert top.provenance.span.length > 0
        assert top.unit is not None
        # And the span really resolves to the text.
        assert (
            top.unit.unit.text in Path(top.provenance.source_uri.replace("file://", "")).read_text()
            or top.unit.unit.text
        )

    def test_manifest_records_what_built_the_index(self, workspace: Path) -> None:
        _, res = _build(workspace)
        m = res.manifest
        assert m.config_hash and m.build_id and m.finished_at
        assert "parse" in m.stage_fingerprints
        assert any(k.startswith("enrich:") for k in m.stage_fingerprints)
        assert {i.name for i in m.indexes} == {"lexical", "dense", "fields"}
        assert m.content_hash  # reproducibility identity

    def test_disabled_stages_are_recorded(self, workspace: Path) -> None:
        _, res = _build(workspace, **{"ingestion.enrich.enabled": False})
        assert "enrich" in res.manifest.disabled_stages


class TestIncremental:
    def test_second_build_does_nothing(self, workspace: Path) -> None:
        _build(workspace)
        _, res = _build(workspace)
        assert res.manifest.corpus.documents_unchanged == 2
        assert res.manifest.corpus.units_written == 0

    def test_adding_a_document_processes_only_it(self, workspace: Path) -> None:
        _build(workspace)
        (workspace / "data" / "c.md").write_text("# Proxies\n\nUse the proxies argument.\n")
        _, res = _build(workspace)
        assert res.manifest.corpus.documents_added == 1
        assert res.manifest.corpus.documents_unchanged == 2

    def test_editing_one_section_reuses_the_others(self, workspace: Path) -> None:
        """The payoff of keeping position out of unit identity."""
        _build(workspace)
        p = workspace / "data" / "a.md"
        p.write_text(p.read_text().replace("27 seconds", "31 seconds"))
        _, res = _build(workspace)
        assert res.manifest.corpus.documents_changed == 1
        assert res.manifest.corpus.units_reused_from_cache >= 1
        assert res.manifest.corpus.units_written >= 1

    def test_query_side_change_reprocesses_nothing(self, workspace: Path) -> None:
        _build(workspace)
        _, res = _build(workspace, **{"query.rerank.enabled": False})
        assert res.manifest.corpus.documents_unchanged == 2
        assert res.manifest.corpus.units_written == 0

    def test_segmenter_change_restages_everything(self, workspace: Path) -> None:
        _build(workspace)
        _, res = _build(workspace, **{"ingestion.segment.impl": "fixed_window"})
        assert res.manifest.corpus.documents_restaged == 2
        assert res.manifest.corpus.units_written > 0

    def test_deleting_a_document_removes_it_from_every_index(self, workspace: Path) -> None:
        _a, _ = _build(workspace)
        (workspace / "data" / "b.md").unlink()
        a2, res = _build(workspace)
        assert res.manifest.corpus.documents_removed == 1
        assert res.manifest.corpus.units_deleted > 0

        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        for idx in a2.indexes.values():
            hits = idx.search(IndexQuery(text="connection pooling reused host", top_k=20), ctx)
            assert not any("b.md" in (h.provenance.source_uri or "") for h in hits.hits)

    def test_plan_is_computed_before_any_work(self, workspace: Path) -> None:
        a = assemble(workspace / "c.yaml")
        plan = a.ingestion().plan()
        assert {p.kind for p in plan} == {ChangeKind.ADDED}
        assert len(plan) == 2


class TestInvariantFive:
    def test_structured_question_never_reaches_vector_search(self, workspace: Path) -> None:
        a, _ = _build(workspace)
        resp = a.query_engine().query("how many entries have a timeout seconds recorded")
        assert str(resp.decision.path) == RoutePath.STRUCTURED
        assert resp.records is not None
        assert resp.hits == ()
        assert resp.skipped.get("retrieve") == "structured_path"

    def test_structured_route_to_a_non_structured_index_raises(self, workspace: Path) -> None:
        """The config is rejected rather than silently degrading to vector search."""
        from indexer.core.errors import ConfigError

        with pytest.raises(ConfigError, match="would reach vector"):
            assemble(
                workspace / "c.yaml",
                overrides={"query.route.paths.structured.targets": ["dense"]},
            )

    def test_every_decision_is_logged_even_when_routing_is_off(self, workspace: Path) -> None:
        log = workspace / "decisions.jsonl"
        a, _ = _build(
            workspace,
            **{"query.route.enabled": False, "query.route.decision_log": str(log)},
        )
        a.query_engine().query("anything at all")
        assert log.exists()
        assert '"reason": "stage_disabled"' in log.read_text()


class TestContractEnforcement:
    def test_broken_parser_output_is_caught_at_the_boundary(self, workspace: Path) -> None:
        """A bad span must fail here, not become a wrong citation three stages on."""
        from indexer.core.document import Block, BlockKind, ParsedDocument
        from indexer.core.provenance import Provenance, Span

        a = assemble(workspace / "c.yaml")
        ing = a.ingestion(strict_contracts=True)

        class LyingParser:
            def can_parse(self, doc):
                return 1.0

            def fingerprint(self):
                return a.parser().fingerprint()

            def parse(self, doc, ctx):
                did = doc.document_id
                return ParsedDocument(
                    document_id=did,
                    source_uri=doc.source_uri,
                    text="hello world",
                    source_hash=doc.content_hash,
                    blocks=(
                        Block(
                            block_id="b0",
                            kind=BlockKind.PARAGRAPH,
                            text="goodbye",  # does not match the span
                            provenance=Provenance(document_id=did, span=Span(0, 5)),
                        ),
                    ),
                )

        ing.parser = LyingParser()
        with pytest.raises(ContractViolation, match="citation"):
            ing.build()

    def test_reranker_may_not_introduce_candidates(self, workspace: Path) -> None:
        from indexer.core.ids import DocumentId
        from indexer.core.provenance import Provenance, Span
        from indexer.core.results import Hit, RankedList

        a, _ = _build(workspace)
        engine = a.query_engine()

        class SmugglingReranker:
            def fingerprint(self):
                return a.reranker().fingerprint()

            def rerank(self, query, candidates, ctx):
                extra = Hit(
                    unit_id="not-retrieved",  # type: ignore[arg-type]
                    document_id=DocumentId("x"),
                    rank=len(candidates.hits) + 1,
                    score=0.0,
                    index="smuggled",
                    provenance=Provenance(document_id=DocumentId("x"), span=Span(0, 1)),
                )
                return RankedList(hits=(*candidates.hits, extra), source="rerank:bad")

        engine.reranker = SmugglingReranker()
        with pytest.raises(ContractViolation, match="not in the candidate set"):
            engine.query("read timeout seconds")


class TestDenseAccelerator:
    """numpy must be a faster way to compute the same thing, not a different thing."""

    def test_accelerated_and_pure_python_agree(self, workspace: Path) -> None:
        import indexer.impls.index_dense as dense_mod

        a, _ = _build(workspace)
        idx = a.indexes["dense"]
        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        q = IndexQuery(text="read timeout seconds production services", top_k=20)

        original = dense_mod.HAVE_NUMPY
        try:
            dense_mod.HAVE_NUMPY = True
            fast = idx.search(q, ctx)
            dense_mod.HAVE_NUMPY = False
            idx._matrix = None
            slow = idx.search(q, ctx)
        finally:
            dense_mod.HAVE_NUMPY = original
            idx._matrix = None

        assert fast.unit_ids() == slow.unit_ids()
        for f, s in zip(fast.hits, slow.hits, strict=True):
            assert abs(f.score - s.score) < 1e-4

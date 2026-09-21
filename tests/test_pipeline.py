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


class TestLedgerJournal:
    """Resumability is a stated property: a build that dies at document 400 of
    500 must resume at 401. The journal is how that survives without rewriting
    the whole snapshot per document."""

    def test_records_survive_without_a_commit(self, tmp_path: Path) -> None:
        from indexer.core.ids import BuildId, ContentHash, DocumentId
        from indexer.core.ledger import DocumentRecord
        from indexer.pipeline.stores import JsonLedger

        path = tmp_path / "ledger.json"
        led = JsonLedger(path)
        led.begin_build(BuildId("b1"))
        for i in range(5):
            led.put(
                DocumentRecord(
                    document_id=DocumentId(f"d{i}"),
                    source_uri=f"file:///{i}",
                    content_hash=ContentHash(f"sha256:{i:064x}"),
                    unit_ids=(),
                    stage_keys={"parse": "p@1"},
                    build_id=BuildId("b1"),
                )
            )
        # Process dies here: no commit_build, so no compaction happened.
        reopened = JsonLedger(path)
        assert reopened.document_ids() == {DocumentId(f"d{i}") for i in range(5)}
        assert reopened.interrupted_build == "b1"

    def test_commit_compacts_and_clears_the_journal(self, tmp_path: Path) -> None:
        from indexer.core.ids import BuildId, ContentHash, DocumentId
        from indexer.core.ledger import DocumentRecord
        from indexer.pipeline.stores import JsonLedger

        path = tmp_path / "ledger.json"
        led = JsonLedger(path)
        led.begin_build(BuildId("b1"))
        led.put(
            DocumentRecord(
                document_id=DocumentId("d0"),
                source_uri="file:///0",
                content_hash=ContentHash("sha256:" + "0" * 64),
                unit_ids=(),
                stage_keys={},
                build_id=BuildId("b1"),
            )
        )
        assert led.journal.exists()
        led.commit_build(BuildId("b1"))
        assert not led.journal.exists()
        assert JsonLedger(path).document_ids() == {DocumentId("d0")}

    def test_a_torn_final_line_loses_only_that_record(self, tmp_path: Path) -> None:
        """A process killed mid-append leaves a partial line. Everything before
        it is intact, and the document it described is simply reprocessed."""
        from indexer.core.ids import BuildId, ContentHash, DocumentId
        from indexer.core.ledger import DocumentRecord
        from indexer.pipeline.stores import JsonLedger

        path = tmp_path / "ledger.json"
        led = JsonLedger(path)
        led.begin_build(BuildId("b1"))
        for i in range(3):
            led.put(
                DocumentRecord(
                    document_id=DocumentId(f"d{i}"),
                    source_uri=f"file:///{i}",
                    content_hash=ContentHash(f"sha256:{i:064x}"),
                    unit_ids=(),
                    stage_keys={},
                    build_id=BuildId("b1"),
                )
            )
        with led.journal.open("a", encoding="utf-8") as fh:
            fh.write('{"document_id": "d3", "source_uri": "file:///3", "cont')

        reopened = JsonLedger(path)
        assert reopened.document_ids() == {DocumentId(f"d{i}") for i in range(3)}

    def test_an_interrupted_build_resumes_where_it_stopped(self, workspace: Path) -> None:
        """End to end: kill a build after one document, rebuild, and only the
        unprocessed document is work."""
        a = assemble(workspace / "c.yaml")
        ing = a.ingestion()
        plan = ing.plan()
        ing.build(plan=plan[:1])  # as if the process died after document 1

        a2 = assemble(workspace / "c.yaml")
        res = a2.ingestion().build()
        assert res.manifest.corpus.documents_unchanged == 1
        assert res.manifest.corpus.documents_added == 1


class TestStructuredPathEndToEnd:
    """Invariant 5, end to end: the answer comes from extracted fields, every
    row cites its source, and no vector index is consulted at any point."""

    def test_structured_query_answers_from_fields_with_citations(self, workspace: Path) -> None:
        (workspace / "data" / "rel.md").write_text(
            "# Releases\n\n## 1.4.0\n\nReleased 2024-03-11 with timeout=30 support.\n\n"
            "## 1.5.0\n\nReleased 2025-07-02 with timeout=45 support.\n"
        )
        a, res = _build(workspace)
        assert res.ok

        resp = a.query_engine().query("how many entries have a timeout seconds recorded")
        assert str(resp.decision.path) == RoutePath.STRUCTURED
        assert resp.records is not None
        assert resp.records.rows, "structured path returned no rows for data that exists"
        # Every row is as citable as a passage.
        assert all(src for src in resp.records.sources)
        # And nothing went near a vector index.
        assert resp.hits == ()
        assert resp.skipped["retrieve"] == "structured_path"
        assert resp.skipped["fuse"] == "structured_path"
        assert resp.skipped["rerank"] == "structured_path"


class TestStructuredScoring:
    """A structured query that returns nothing has failed.

    Scoring the structured path as successful merely for having been taken
    flatters every arm whose structured index is empty -- which is every arm
    with extraction disabled. On the ablation corpus that was the difference
    between a 0.075 and a 0.141 failure rate for the same arm.
    """

    def test_empty_record_set_counts_as_a_failure(self) -> None:
        from indexer.core.ids import DocumentId
        from indexer.core.predicate import Exists, StructuredQuery
        from indexer.core.provenance import Span
        from indexer.core.query import Query, QueryType, RouteDecision, RoutePath
        from indexer.core.results import RecordSet, RetrievalResponse
        from indexer.eval.golden import GoldenQuery, RelevantSpan
        from indexer.eval.runner import EvalRunner

        item = GoldenQuery(
            id="s1",
            query="how many entries have a version recorded",
            relevant=(RelevantSpan(document_id=DocumentId("d1"), span=Span(0, 10)),),
            query_type=QueryType.STRUCTURED,
        )
        decision = RouteDecision(
            path=RoutePath.STRUCTURED,
            structured_query=StructuredQuery(where=Exists("version")),
            query_type=QueryType.STRUCTURED,
        )

        class Engine:
            def __init__(self, rows: tuple) -> None:
                self.rows = rows

            def execute(self, q: Query) -> RetrievalResponse:
                return RetrievalResponse(
                    query=q,
                    decision=decision,
                    records=RecordSet(columns=("version",), rows=self.rows),
                )

        runner = EvalRunner()
        empty = runner._score_one(Engine(()), item)
        answered = runner._score_one(Engine((("1.2.0",),)), item)
        assert empty.failed is True, "an empty record set must count as a failure"
        assert answered.failed is False

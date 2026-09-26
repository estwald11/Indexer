"""Tests for the frame's own guarantees.

These are not tests of any implementation -- there are none yet. They test the
properties the contracts claim, using hand-built stage output. If one of these
fails, a promise in ``ARCHITECTURE.md`` has stopped being true.
"""

from __future__ import annotations

import pytest

from indexer.core import (
    Block,
    BlockKind,
    ContextScope,
    DocumentId,
    EnrichedUnit,
    Enrichment,
    Hit,
    ParsedDocument,
    Provenance,
    RankedList,
    Span,
    StageFingerprint,
    Unit,
    cache_key,
    diff_units,
    hash_obj,
    hash_text,
    make_unit_id,
)
from indexer.eval.checks import check_parsed_document, check_ranked_list


def _doc(text: str, blocks: list[tuple[str, int, int]]) -> ParsedDocument:
    did = DocumentId("d1")
    return ParsedDocument(
        document_id=did,
        source_uri="file:///d1.md",
        text=text,
        source_hash=hash_text(text),
        blocks=tuple(
            Block(
                block_id=f"b{i}",
                kind=BlockKind.PARAGRAPH,
                text=t,
                provenance=Provenance(document_id=did, span=Span(s, e)),
            )
            for i, (t, s, e) in enumerate(blocks)
        ),
    )


class TestProvenanceIsCheckable:
    """The parse contract's span equality is the root of every citation."""

    def test_consistent_document_passes(self) -> None:
        text = "First para.\n\nSecond para."
        doc = _doc(text, [("First para.", 0, 11), ("Second para.", 13, 25)])
        assert check_parsed_document(doc) == []

    def test_wrong_span_is_caught(self) -> None:
        text = "First para.\n\nSecond para."
        doc = _doc(text, [("First para.", 0, 11), ("Second para.", 12, 24)])
        problems = check_parsed_document(doc)
        assert len(problems) == 1
        assert "citation" in problems[0]

    def test_out_of_order_blocks_are_caught(self) -> None:
        text = "aaa bbb"
        doc = _doc(text, [("bbb", 4, 7), ("aaa", 0, 3)])
        assert any("ascending" in p for p in check_parsed_document(doc))

    def test_table_without_structure_is_caught(self) -> None:
        did = DocumentId("d1")
        text = "| a | b |"
        doc = ParsedDocument(
            document_id=did,
            source_uri="",
            text=text,
            source_hash=hash_text(text),
            blocks=(
                Block(
                    block_id="b0",
                    kind=BlockKind.TABLE,
                    text=text,
                    provenance=Provenance(document_id=did, span=Span(0, 9)),
                    table=None,
                ),
            ),
        )
        assert any("structured path" in p for p in check_parsed_document(doc))


class TestUnitIdentityIsPositionIndependent:
    """The incremental guarantee rests on this."""

    def test_same_content_same_id(self) -> None:
        a = make_unit_id(DocumentId("d1"), hash_text("hello"))
        b = make_unit_id(DocumentId("d1"), hash_text("hello"))
        assert a == b

    def test_different_document_different_id(self) -> None:
        a = make_unit_id(DocumentId("d1"), hash_text("hello"))
        b = make_unit_id(DocumentId("d2"), hash_text("hello"))
        assert a != b

    def test_repeated_text_disambiguated_by_occurrence(self) -> None:
        a = make_unit_id(DocumentId("d1"), hash_text("Page 1"), occurrence=0)
        b = make_unit_id(DocumentId("d1"), hash_text("Page 1"), occurrence=1)
        assert a != b

    def test_inserting_a_unit_does_not_renumber_the_rest(self) -> None:
        """The property that makes a one-line edit cost one unit, not a document."""
        did = DocumentId("d1")
        before = [make_unit_id(did, hash_text(t)) for t in ("alpha", "beta", "gamma")]
        after = [make_unit_id(did, hash_text(t)) for t in ("alpha", "NEW", "beta", "gamma")]
        to_upsert, to_delete = diff_units(before, after)
        assert len(to_upsert) == 1
        assert to_delete == ()


class TestIndexingSurfaceIsShared:
    """Invariant 3: context is prepended before *both* dense and lexical."""

    def _unit(self) -> Unit:
        return Unit(
            unit_id=make_unit_id(DocumentId("d1"), hash_text("body text")),
            document_id=DocumentId("d1"),
            text="body text",
            provenance=Provenance(document_id=DocumentId("d1"), span=Span(0, 9)),
        )

    def test_no_enrichment_degrades_to_raw_text(self) -> None:
        eu = EnrichedUnit(unit=self._unit())
        assert eu.indexing_text() == "body text"

    def test_context_is_prepended(self) -> None:
        eu = EnrichedUnit(unit=self._unit()).with_enrichment(
            Enrichment(enricher="ctx", fingerprint="f", context="From the 2023 filing.")
        )
        assert eu.indexing_text() == "From the 2023 filing.\n\nbody text"

    def test_multiple_contexts_are_order_stable(self) -> None:
        """Dict order must not leak into a hashed string."""
        base = EnrichedUnit(unit=self._unit())
        a = base.with_enrichment(Enrichment(enricher="a", fingerprint="1", context="AAA"))
        a = a.with_enrichment(Enrichment(enricher="z", fingerprint="1", context="ZZZ"))
        b = base.with_enrichment(Enrichment(enricher="z", fingerprint="1", context="ZZZ"))
        b = b.with_enrichment(Enrichment(enricher="a", fingerprint="1", context="AAA"))
        assert a.indexing_text() == b.indexing_text()
        assert a.indexing_hash == b.indexing_hash

    def test_enrichment_is_idempotent(self) -> None:
        """Running an enricher twice replaces, never duplicates."""
        eu = EnrichedUnit(unit=self._unit())
        e = Enrichment(enricher="ctx", fingerprint="f", context="ONCE")
        assert eu.with_enrichment(e).with_enrichment(e).indexing_text() == (
            eu.with_enrichment(e).indexing_text()
        )


class TestCacheKeys:
    def _fp(self, **kw: str) -> StageFingerprint:
        d = {"stage": "enrich", "impl": "ctx", "version": "1", "params_hash": "p"}
        d.update(kw)
        return StageFingerprint(**d)  # type: ignore[arg-type]

    def test_version_bump_invalidates(self) -> None:
        """An edited prompt without a version bump serves stale results forever."""
        assert cache_key(self._fp(), "in") != cache_key(self._fp(version="2"), "in")

    def test_params_change_invalidates(self) -> None:
        assert cache_key(self._fp(), "in") != cache_key(self._fp(params_hash="q"), "in")

    def test_scope_hash_separates_unit_and_document_scope(self) -> None:
        """A unit-scoped enricher survives an edit elsewhere in its document."""
        unit_scoped = cache_key(self._fp(), "unit-hash", scope_hash="")
        doc_scoped_v1 = cache_key(self._fp(), "unit-hash", scope_hash="docv1")
        doc_scoped_v2 = cache_key(self._fp(), "unit-hash", scope_hash="docv2")
        assert doc_scoped_v1 != doc_scoped_v2
        assert unit_scoped not in (doc_scoped_v1, doc_scoped_v2)

    def test_stages_do_not_collide(self) -> None:
        a = cache_key(self._fp(stage="parse"), "x")
        b = cache_key(self._fp(stage="segment"), "x")
        assert a != b

    def test_canonical_hash_ignores_key_order(self) -> None:
        """Reordering a YAML file must not invalidate an index."""
        assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})


class TestRankedListDiscipline:
    def _hit(self, uid: str, rank: int) -> Hit:
        return Hit(
            unit_id=uid,  # type: ignore[arg-type]
            document_id=DocumentId("d1"),
            rank=rank,
            score=1.0 / rank,
            index="lexical",
            provenance=Provenance(document_id=DocumentId("d1"), span=Span(0, 5)),
        )

    def test_sparse_ranks_are_rejected(self) -> None:
        """Fusion is rank-based; a gap would silently distort every RRF score."""
        with pytest.raises(ValueError, match="densely ranked"):
            RankedList(hits=(self._hit("u1", 1), self._hit("u2", 3)), source="lexical")

    def test_duplicate_units_are_caught(self) -> None:
        rl = RankedList(hits=(self._hit("u1", 1), self._hit("u1", 2)), source="lexical")
        assert any("duplicate" in p for p in check_ranked_list(rl))

    def test_from_scored_assigns_dense_ranks(self) -> None:
        rl = RankedList.from_scored(
            [(self._hit("u1", 1), 0.2), (self._hit("u2", 1), 0.9)], source="dense"
        )
        assert [h.rank for h in rl.hits] == [1, 2]
        assert rl.hits[0].unit_id == "u2"


class TestScopeDeclaration:
    def test_scopes_are_ordered_by_blast_radius(self) -> None:
        assert ContextScope.UNIT != ContextScope.DOCUMENT
        assert [s.value for s in ContextScope] == [
            "unit",
            "neighbors",
            "document",
            "related",
            "corpus",
        ]


class TestContextSpecificity:
    """Invariant 3's benefit needs context that distinguishes one chunk from another.

    Context that repeats across a document's units cannot help rank one above
    another, while it does dilute the terms that can. The symptom --
    contextualisation not helping -- looks exactly like the enricher being
    mis-wired, and the two have different fixes, so this is measured.
    """

    def _units(self, contexts: list[str], bodies: list[str]) -> list[EnrichedUnit]:
        did = DocumentId("d1")
        out = []
        for i, (ctx, body) in enumerate(zip(contexts, bodies, strict=True)):
            u = Unit(
                unit_id=make_unit_id(did, hash_text(body), occurrence=i),
                document_id=did,
                text=body,
                provenance=Provenance(document_id=did, span=Span(0, len(body))),
            )
            out.append(
                EnrichedUnit(unit=u).with_enrichment(
                    Enrichment(enricher="ctx", fingerprint="1", context=ctx)
                )
            )
        return out

    def test_identical_context_is_flagged(self) -> None:
        from indexer.eval.checks import check_context_specificity

        boiler = "The Acme Toolkit is a library for widgets. Topics: widgets, gears, bolts."
        units = self._units(
            [boiler] * 4,
            ["alpha content here", "beta content here", "gamma content", "delta content"],
        )
        problems = check_context_specificity(units)
        assert problems
        assert "identical between adjacent units" in problems[0]

    def test_chunk_specific_context_passes(self) -> None:
        from indexer.eval.checks import check_context_specificity

        units = self._units(
            [
                "Acme Toolkit, installation on Windows",
                "Acme Toolkit, configuring the scheduler",
                "Acme Toolkit, migration from version two",
                "Acme Toolkit, troubleshooting network timeouts",
            ],
            ["alpha content here", "beta content here", "gamma content", "delta content"],
        )
        assert check_context_specificity(units) == []

    def test_no_context_is_not_an_error(self) -> None:
        """Contextualisation off is an ablation arm, not a violation."""
        from indexer.eval.checks import check_context_specificity

        did = DocumentId("d1")
        units = [
            EnrichedUnit(
                unit=Unit(
                    unit_id=make_unit_id(did, hash_text(t)),
                    document_id=did,
                    text=t,
                    provenance=Provenance(document_id=did, span=Span(0, len(t))),
                )
            )
            for t in ("one body", "two body", "three body")
        ]
        assert check_context_specificity(units) == []


class TestErrorsAreVisible:
    """Metrics are averaged over the queries that ran. An arm where most queries
    raised therefore reports healthy numbers over its survivors, which reads as
    a good result rather than a broken one."""

    def test_delta_table_names_a_broken_arm(self) -> None:
        from indexer.eval.harness import AblationResult
        from indexer.eval.metrics import RunReport

        r = AblationResult(
            reports=[
                RunReport(arm="ok", n_queries=100, precision={5: 0.4}, retrieval_failure_rate=0.1),
                RunReport(
                    arm="broken",
                    n_queries=100,
                    errors=93,
                    precision={5: 0.9},  # over the 7 that ran
                    retrieval_failure_rate=0.0,
                ),
            ],
            baseline_arm="ok",
        )
        out = r.delta_table()
        assert "93/100 queries raised" in out
        assert "not comparable" in out
        assert "err" in out.splitlines()[0]

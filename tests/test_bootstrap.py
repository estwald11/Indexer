"""Golden-set generation, and the one ordering mistake that would void it.

The paraphrasing generator's whole value depends on screening answerability
*before* rewriting. Screened afterwards with a lexical baseline, every item that
requires matching meaning is rejected for being unfindable by exact match -- the
set looks rigorously filtered and has removed exactly what it existed to add.
That is what most of this file is about.
"""

from __future__ import annotations

from typing import Any

import pytest

from indexer.core.document import Block, BlockKind, ParsedDocument
from indexer.core.ids import DocumentId, UnitId, hash_text
from indexer.core.provenance import Provenance, Span
from indexer.core.query import QueryType
from indexer.core.registry import resolve
from indexer.core.unit import EnrichedUnit, Unit
from indexer.eval.bootstrap import HeuristicBootstrapper, LLMBootstrapper, measure_overlap
from indexer.eval.golden import GoldenQuery, GoldenSet, RelevantSpan, load_golden_set

PASSAGES = [
    "The read timeout controls how long the client waits for the server to send "
    "a response byte. Set it with the timeout keyword argument on any request.",
    "Connection pooling reuses open sockets across requests to the same host, "
    "which removes the handshake cost from every call after the first.",
    "Retries repeat a failed request using exponential backoff, doubling the "
    "wait between attempts until the ceiling configured on the adapter.",
    "Session objects persist cookies and connection pools between calls, so "
    "logging in once carries across every later request made through them.",
]


def _doc(doc_id: str = "d1") -> ParsedDocument:
    text = "\n\n".join(PASSAGES)
    blocks, off = [], 0
    for p in PASSAGES:
        blocks.append(
            Block(
                block_id=f"b{len(blocks)}",
                kind=BlockKind.PARAGRAPH,
                text=p,
                provenance=Provenance(
                    document_id=DocumentId(doc_id),
                    span=Span(off, off + len(p)),
                    source_uri="mem://d1",
                ),
            )
        )
        off += len(p) + 2
    return ParsedDocument(
        document_id=DocumentId(doc_id),
        source_uri="mem://d1",
        text=text,
        blocks=tuple(blocks),
        source_hash=hash_text(text),
    )


def _units(doc: ParsedDocument) -> list[EnrichedUnit]:
    return [
        EnrichedUnit(
            unit=Unit(
                unit_id=UnitId(f"u{i}"),
                document_id=doc.document_id,
                text=b.text,
                provenance=Provenance(
                    document_id=doc.document_id,
                    span=b.provenance.span,
                    source_uri=doc.source_uri,
                ),
                section_path=("Requests",),
            ),
            enrichments={},
        )
        for i, b in enumerate(doc.blocks)
    ]


class _StubClient:
    """Stands in for the Anthropic client, and records what it was asked."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.messages = self

    def create(self, **kw: Any) -> Any:
        self.prompts.append(kw["messages"][0]["content"])
        text = self.replies.pop(0) if self.replies else "a rewritten question"

        block = type("T", (), {"text": text})()
        return type(
            "_R",
            (),
            {
                "content": [block],
                "usage": type("U", (), {"input_tokens": 11, "output_tokens": 7})(),
            },
        )()


class TestMeasureOverlap:
    """The number that says what a golden set can measure."""

    def test_verbatim_query_scores_one(self) -> None:
        assert measure_overlap("read timeout server", PASSAGES[0]) == 1.0

    def test_disjoint_vocabulary_scores_zero(self) -> None:
        assert measure_overlap("zebra pancake xylophone", PASSAGES[0]) == 0.0

    def test_stopwords_do_not_inflate_it(self) -> None:
        assert measure_overlap("the and of zebra", PASSAGES[0]) == 0.0

    def test_no_content_words_is_not_a_number(self) -> None:
        got = measure_overlap("the and of", PASSAGES[0])
        assert got != got  # NaN: "cannot tell", not "no overlap"


class TestHeuristicRecordsWhatItCosts:
    def test_items_carry_their_overlap(self) -> None:
        doc = _doc()
        gs = HeuristicBootstrapper({"detail_terms": 3, "min_unit_chars": 50}).bootstrap(
            [doc], _units(doc)
        )
        assert gs.queries
        assert all(q.lexical_overlap is not None for q in gs.queries)
        # This generator lifts its detail terms from the unit, so the overlap it
        # records should be high. If this ever drops, the set got easier for
        # dense retrieval without anyone deciding that it should.
        assert gs.mean_lexical_overlap() > 0.6

    def test_set_reports_overlap_and_paraphrase_counts(self) -> None:
        doc = _doc()
        gs = HeuristicBootstrapper({"min_unit_chars": 50}).bootstrap([doc], _units(doc))
        st = gs.stats()
        assert st["paraphrased"] == 0
        assert 0.0 <= st["lexical_overlap"] <= 1.0


class TestLLMBootstrapper:
    def _run(self, replies: list[str], params: dict[str, Any] | None = None) -> Any:
        doc = _doc()
        client = _StubClient(replies)
        boot = LLMBootstrapper({"min_unit_chars": 50, **(params or {})}, client=client)
        return boot.bootstrap([doc], _units(doc)), client

    def test_rewrites_are_what_ships_and_the_literal_is_kept(self) -> None:
        gs, _ = self._run(["how long before a stalled download gives up"] * 8)
        rewritten = [q for q in gs.queries if q.paraphrased]
        assert rewritten
        for q in rewritten:
            assert q.query == "how long before a stalled download gives up"
            assert q.source_query and q.source_query != q.query
            assert "paraphrase" in q.tags

    def test_rewriting_lowers_the_overlap_it_exists_to_lower(self) -> None:
        doc = _doc()
        literal = HeuristicBootstrapper({"min_unit_chars": 50}).bootstrap([doc], _units(doc))
        gs, _ = self._run(["how long before a stalled download gives up"] * 8)
        assert gs.mean_lexical_overlap() < literal.mean_lexical_overlap()

    def test_the_passage_is_shown_and_the_subject_is_kept(self) -> None:
        _, client = self._run(["rewritten"] * 8)
        assert client.prompts
        assert any(PASSAGES[0][:40] in p for p in client.prompts)
        assert all("Avoid the passage's distinctive wording" in p for p in client.prompts)

    def test_an_echoed_rewrite_is_labelled_not_hidden(self) -> None:
        """A model that restates the passage has failed, and silently keeping
        the result would refill the set with the items rewriting removes."""
        gs, _ = self._run(["read timeout controls how long the client waits"] * 8)
        echoed = [q for q in gs.queries if "paraphrase_echoed" in q.tags]
        assert echoed
        assert "echoed" in gs.notes

    def test_failed_rewrites_can_be_dropped_instead(self) -> None:
        kept, _ = self._run(["read timeout controls how long the client waits"] * 8)
        dropped, _ = self._run(
            ["read timeout controls how long the client waits"] * 8,
            {"drop_failed_rewrites": True},
        )
        assert len(dropped.queries) < len(kept.queries)

    def test_an_empty_reply_leaves_the_literal_item_intact(self) -> None:
        gs, _ = self._run([""] * 8)
        assert gs.queries
        assert not any(q.paraphrased for q in gs.queries)

    def test_gold_spans_survive_the_rewrite(self) -> None:
        """Only the wording changes, so the two sets stay comparable arm for arm."""
        doc = _doc()
        literal = HeuristicBootstrapper({"min_unit_chars": 50}).bootstrap([doc], _units(doc))
        gs, _ = self._run(["a rewritten question about waiting"] * 8)
        assert [q.relevant for q in gs.queries] == [q.relevant for q in literal.queries]

    def test_prompt_is_in_the_fingerprint(self) -> None:
        """An edited prompt changes every query; a generator hash that did not
        move would claim comparability the set does not have."""
        a = LLMBootstrapper({}, client=_StubClient([]))
        b = LLMBootstrapper({}, client=_StubClient([]))
        b.PROMPT = a.PROMPT + " and be terse"  # type: ignore[misc]
        assert a.fingerprint().key() != b.fingerprint().key()

    def test_missing_dependency_names_the_package(self) -> None:
        doc = _doc()
        with pytest.raises(RuntimeError, match="anthropic"):
            LLMBootstrapper({}).bootstrap([doc], _units(doc))

    def test_registered_with_its_requirement(self) -> None:
        reg = resolve("bootstrap", "llm_bootstrap")
        assert reg.requires == ("anthropic",)


class TestGoldenRoundTrip:
    def test_new_fields_survive_disk(self, tmp_path: Any) -> None:
        from indexer.eval.golden import write_golden_set

        gs = GoldenSet(
            queries=(
                GoldenQuery(
                    id="q1",
                    query="how long before a stalled download gives up",
                    source_query="requests read timeout seconds",
                    lexical_overlap=0.25,
                    relevant=(
                        RelevantSpan(
                            document_id=DocumentId("d1"), span=Span(0, 10), snippet="read timeout"
                        ),
                    ),
                    query_type=QueryType.FACTUAL,
                ),
            )
        )
        path = tmp_path / "g.jsonl"
        write_golden_set(gs, path)
        back = load_golden_set(path)
        q = back.queries[0]
        assert q.source_query == "requests read timeout seconds"
        assert q.lexical_overlap == 0.25
        assert q.paraphrased

    def test_a_set_without_the_field_falls_back_to_the_snippet(self, tmp_path: Any) -> None:
        """Sets written before `lexical_overlap` existed still report one."""
        path = tmp_path / "old.jsonl"
        path.write_text(
            '{"id":"q1","query":"read timeout seconds","relevant":'
            '[{"document_id":"d1","span":[0,10],"snippet":"the read timeout"}]}\n'
        )
        back = load_golden_set(path)
        assert back.queries[0].lexical_overlap is None
        assert back.queries[0].overlap() == pytest.approx(2 / 3)  # read, timeout of 3 terms

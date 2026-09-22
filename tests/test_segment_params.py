"""Segmenter size parameters do what the config says.

``min_tokens`` and ``overlap_tokens`` were accepted by the schema, passed to the
segmenter by the assembler, and ignored; tables were always split at 512 tokens
whatever ``max_tokens`` said. A parameter that is validated and does nothing is
worse than one that is rejected: an ablation over it reports no effect, and the
report is believed.
"""

from __future__ import annotations

from itertools import pairwise

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.document import BlockKind, SourceDocument
from indexer.core.ids import DocumentId, hash_text
from indexer.core.stages import StageContext
from indexer.eval.checks import check_units
from indexer.impls.parse import MarkdownParser
from indexer.impls.segment import StructuralSegmenter, estimate_tokens

CTX = StageContext(cache=NullCache(), accountant=InMemoryAccountant())


class _Doc(SourceDocument):
    __slots__ = ()

    def load(self) -> bytes:
        return self.metadata["body"].encode()  # type: ignore[no-any-return]


def _parse(body: str):  # type: ignore[no-untyped-def]
    doc = _Doc(
        document_id=DocumentId("d"),
        source_uri="mem://d",
        content_hash=hash_text(body),
        media_type="text/markdown",
        size_bytes=len(body),
        metadata={"body": body},
    )
    return MarkdownParser({}).parse(doc, CTX)


def _segment(body: str, **params):  # type: ignore[no-untyped-def]
    parsed = _parse(body)
    units = list(StructuralSegmenter({"merge_below_tokens": 0, **params}).segment(parsed, CTX))
    assert check_units(units, parsed) == []
    return units


# One paragraph of ~60 sentences, each ~10 tokens: forced to split by size.
LONG = " ".join(f"Sentence number {i} says something about invoices." for i in range(60))


class TestTables:
    def test_tables_split_at_the_configured_limit(self) -> None:
        rows = "\n".join(f"| row {i} | value {i} | note {i} |" for i in range(80))
        body = f"# T\n\n| a | b | c |\n|---|---|---|\n{rows}\n"
        big = _segment(body, max_tokens=512)
        small = _segment(body, max_tokens=100)
        # The limit used to be a hard-coded 512 whatever max_tokens said.
        assert len(small) > len(big)
        for u in small:
            assert estimate_tokens(u.text) <= 100 + 20  # header repetition slack
            assert u.text.startswith("| a | b | c |")


class TestMinTokens:
    def test_a_split_never_leaves_a_tiny_tail(self) -> None:
        body = f"# T\n\n{LONG} Tail.\n"
        loose = _segment(body, max_tokens=100, min_tokens=0)
        tight = _segment(body, max_tokens=100, min_tokens=60)
        assert estimate_tokens(tight[-1].text) >= 60
        assert len(tight) <= len(loose)


class TestOverlap:
    def test_split_pieces_repeat_the_end_of_the_one_before(self) -> None:
        body = f"# T\n\n{LONG}\n"
        units = _segment(body, max_tokens=100, overlap_tokens=25)
        assert len(units) > 2
        for prev, nxt in pairwise(units):
            a, b = prev.provenance.span, nxt.provenance.span
            assert b.start < a.end, "consecutive pieces must overlap"
            assert a.end - b.start <= 25 * 4 + 60  # within budget, at a sentence edge

    def test_no_overlap_by_default(self) -> None:
        units = _segment(f"# T\n\n{LONG}\n", max_tokens=100)
        for prev, nxt in pairwise(units):
            assert nxt.provenance.span.start >= prev.provenance.span.end

    def test_block_runs_carry_whole_blocks(self) -> None:
        paras = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(8))
        units = _segment(f"# T\n\n{paras}\n", max_tokens=120, overlap_tokens=60)
        assert len(units) > 1
        # The second unit starts with the paragraph the first one ended with.
        last_para_of_first = units[0].text.split("\n\n")[-1]
        assert units[1].text.startswith(last_para_of_first)


class TestItalianSentences:
    def test_sentences_opening_with_an_accented_capital_split(self) -> None:
        # Every sentence opens with "È": the only boundaries there are. With an
        # ASCII-capital rule none was found and the block stayed one piece,
        # four times the cap.
        text = " ".join(f"È importante notare il punto {i} del contratto." for i in range(40))
        units = _segment(f"# T\n\n{text}\n", max_tokens=80)
        assert len(units) > 1
        # Within the cap, up to the rounding of per-sentence estimates.
        assert all(estimate_tokens(u.text) <= 88 for u in units)
        assert all(str(u.kind) != BlockKind.HEADING for u in units)

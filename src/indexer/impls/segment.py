"""Segmenters: structural, fixed-window, and whole-document.

The contract says boundaries come from structure, not character counts. Three
implementations make that testable rather than asserted:

``structural``
    Honours the contract. Sections are the unit; oversized sections split at
    block boundaries; tables split by row groups with headers repeated.

``fixed_window``
    Deliberately violates the spirit of it. It exists because "structure-aware
    chunking beats fixed windows" is a claim, and a claim needs a control arm.

``whole_document``
    One unit per document. The disabled-segment behaviour, and a real baseline:
    on short documents it sometimes wins.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from indexer.core.document import Block, BlockKind, ParsedDocument, Table
from indexer.core.ids import hash_text, make_unit_id
from indexer.core.provenance import Provenance, Span
from indexer.core.registry import register
from indexer.core.stages import StageContext
from indexer.core.unit import Unit, UnitKind
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["FixedWindowSegmenter", "StructuralSegmenter", "WholeDocumentSegmenter"]


def estimate_tokens(text: str) -> int:
    """Rough token count without a tokenizer.

    ~4 characters per token for English prose. Deliberately approximate: these
    are *limits*, and spending a tokenizer dependency to make a ceiling precise
    would be paying for accuracy that changes no decision.
    """
    return max(1, len(text) // 4)


class _UnitBuilder:
    """Assembles units and keeps identity, ordering and provenance consistent."""

    __slots__ = ("_occurrences", "doc", "units")

    def __init__(self, doc: ParsedDocument) -> None:
        self.doc = doc
        self.units: list[Unit] = []
        self._occurrences: dict[str, int] = {}

    def add(
        self,
        text: str,
        blocks: Sequence[Block],
        section_path: tuple[str, ...],
        kind: UnitKind | str = UnitKind.PROSE,
        table_ref: str | None = None,
        span: Span | None = None,
        verbatim: bool = True,
    ) -> None:
        if not text.strip():
            return
        h = hash_text(text)
        # `occurrence` disambiguates genuinely identical text within a document
        # (repeated boilerplate, a repeated table header) without letting
        # position into the id of anything else.
        occ = self._occurrences.get(h, 0)
        self._occurrences[h] = occ + 1
        prov = blocks[0].provenance
        for b in blocks[1:]:
            prov = prov.merged_with(b.provenance)
        unit_span = span or prov.span
        self.units.append(
            Unit(
                unit_id=make_unit_id(self.doc.document_id, h, occurrence=occ),
                document_id=self.doc.document_id,
                text=text,
                provenance=Provenance(
                    document_id=self.doc.document_id,
                    span=unit_span,
                    source_uri=self.doc.source_uri,
                    pages=prov.pages,
                    bbox=prov.bbox,
                    block_ids=tuple(b.block_id for b in blocks),
                ),
                section_path=section_path,
                kind=kind,
                ordinal=len(self.units),
                table_ref=table_ref,
                verbatim=verbatim,
                metadata=dict(self.doc.metadata),
            )
        )

    def merge_forward(self, last: Unit, extra: Sequence[Block], parsed: ParsedDocument) -> None:
        """Extend a unit to cover following blocks, as one contiguous slice.

        Text comes from the canonical text, not from re-joining block texts, so
        the span invariant holds by construction for whatever lies between.
        """
        start = last.provenance.span.start
        end = extra[-1].provenance.span.end
        text = parsed.text[start:end]
        covered = tuple(
            blk.block_id
            for blk in parsed.blocks
            if blk.provenance.span.start >= start and blk.provenance.span.end <= end
        )
        h = hash_text(text)
        occ = self._occurrences.get(h, 0)
        self._occurrences[h] = occ + 1
        self.units.append(
            Unit(
                unit_id=make_unit_id(self.doc.document_id, h, occurrence=occ),
                document_id=self.doc.document_id,
                text=text,
                provenance=Provenance(
                    document_id=self.doc.document_id,
                    span=Span(start, end),
                    source_uri=self.doc.source_uri,
                    pages=last.provenance.pages,
                    block_ids=covered,
                ),
                section_path=last.section_path,
                kind=last.kind,
                ordinal=len(self.units),
                metadata=dict(self.doc.metadata),
            )
        )

    def finish(self) -> list[Unit]:
        """Link neighbours. Done at the end because a unit cannot know its
        successor while being built, and enrichers want the window."""
        from dataclasses import replace

        out: list[Unit] = []
        for i, u in enumerate(self.units):
            out.append(
                replace(
                    u,
                    prev_unit_id=self.units[i - 1].unit_id if i else None,
                    next_unit_id=(self.units[i + 1].unit_id if i + 1 < len(self.units) else None),
                )
            )
        return out


# --------------------------------------------------------------------------- #
# structural                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StructuralParams:
    max_tokens: int = 512
    min_tokens: int = 64
    overlap_tokens: int = 0
    #: Merge a section under this size into the next one. Without it a document
    #: of many one-line headings produces a unit per heading, each too small to
    #: retrieve on and each costing an embedding.
    merge_below_tokens: int = 32
    keep_code: bool = True


@register(
    "segment",
    "structural",
    version="2",
    params_model=dataclass_params(StructuralParams),
    summary="Sections are units. Splits at block boundaries; never splits a table row.",
)
def _make_structural(params: dict[str, Any], **_: Any) -> StructuralSegmenter:
    return StructuralSegmenter(params)


class StructuralSegmenter(StageImpl):
    """The contract-honouring segmenter.

    Boundary rule, in order of precedence:
      1. A heading opens a new section.
      2. A section over ``max_tokens`` splits at block boundaries, never inside
         a block.
      3. A single block over ``max_tokens`` splits at sentence boundaries -- the
         last resort, and the only place character counts get a vote.
      4. A table splits by row groups, repeating its header rows, so every piece
         stays independently interpretable.

    The property this buys, checked by ``check_unit_stability``: editing section
    3 does not change the units of section 4.
    """

    STAGE, IMPL, VERSION = "segment", "structural", "2"

    def segment(self, parsed: ParsedDocument, ctx: StageContext) -> Sequence[Unit]:
        max_tok = int(self.param("max_tokens", 512))
        merge_below = int(self.param("merge_below_tokens", 32))
        b = _UnitBuilder(parsed)

        for path, blocks in _sections(parsed):
            content = [blk for blk in blocks if str(blk.kind) != BlockKind.HEADING]
            if not content:
                continue

            for group in _split_tables(content):
                if isinstance(group, tuple):  # (table_block, row_groups)
                    tblock, chunks = group
                    split = len(chunks) > 1
                    for ci, chunk_text in enumerate(chunks):
                        # A split table repeats its header rows into every
                        # piece, so the text is derived from the span rather
                        # than copied from it. Declared, not tolerated.
                        b.add(
                            chunk_text,
                            [tblock],
                            path,
                            kind=UnitKind.TABLE_ROWS if split else UnitKind.TABLE,
                            table_ref=f"{tblock.block_id}#{ci}",
                            verbatim=not split,
                        )
                    continue

                pending: list[Block] = []
                pending_tokens = 0
                for blk in group:
                    tok = estimate_tokens(blk.text)
                    if tok > max_tok:
                        if pending:
                            b.add(_join(pending), pending, path)
                            pending, pending_tokens = [], 0
                        base = blk.provenance.span.start
                        for rel_start, rel_end in _split_long_block(blk.text, max_tok):
                            # Slice the block's own text rather than re-joining
                            # sentences: re-joining normalises whitespace, and
                            # the unit then no longer matches the span it cites.
                            b.add(
                                blk.text[rel_start:rel_end],
                                [blk],
                                path,
                                kind=_kind_of(blk),
                                span=Span(base + rel_start, base + rel_end),
                            )
                        continue
                    if pending and pending_tokens + tok > max_tok:
                        b.add(_join(pending), pending, path)
                        pending, pending_tokens = [], 0
                    pending.append(blk)
                    pending_tokens += tok
                if pending:
                    if pending_tokens < merge_below and b.units:
                        # Too small to retrieve on and not worth an embedding, so
                        # fold it into the previous unit.
                        #
                        # The merged unit is taken as a *contiguous slice* of the
                        # canonical text rather than by re-joining block texts.
                        # Joining would silently drop whatever sits between the
                        # two runs -- the intervening heading -- while the span
                        # still covered it, breaking `text == parsed.text[span]`
                        # and with it every citation from this unit. Slicing also
                        # keeps that heading, which is context worth having.
                        last = b.units.pop()
                        b.merge_forward(last, pending, parsed)
                    else:
                        b.add(_join(pending), pending, path)
        return b.finish()


def _kind_of(blk: Block) -> UnitKind | str:
    return {
        BlockKind.CODE: UnitKind.CODE,
        BlockKind.TABLE: UnitKind.TABLE,
        BlockKind.FIGURE: UnitKind.FIGURE,
        BlockKind.LIST_ITEM: UnitKind.LIST,
    }.get(
        BlockKind(str(blk.kind)) if str(blk.kind) in set(BlockKind) else BlockKind.OTHER,
        UnitKind.PROSE,
    )


def _join(blocks: Sequence[Block]) -> str:
    return "\n\n".join(b.text for b in blocks)


def _sections(parsed: ParsedDocument) -> list[tuple[tuple[str, ...], list[Block]]]:
    """Group blocks under their heading trail, preserving reading order."""
    out: list[tuple[tuple[str, ...], list[Block]]] = []
    stack: list[tuple[int, str]] = []
    current: list[Block] = []
    path: tuple[str, ...] = ()

    for blk in parsed.blocks:
        if str(blk.kind) == BlockKind.HEADING:
            if current:
                out.append((path, current))
                current = []
            level = blk.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, blk.text))
            path = tuple(t for _, t in stack)
            continue
        current.append(blk)
    if current:
        out.append((path, current))
    return out


def _split_tables(blocks: list[Block]) -> list[Any]:
    """Separate tables from prose runs so a table is never merged into a chunk."""
    out: list[Any] = []
    run: list[Block] = []
    for blk in blocks:
        if str(blk.kind) == BlockKind.TABLE and blk.table is not None:
            if run:
                out.append(run)
                run = []
            out.append((blk, _split_table(blk.text, blk.table)))
        else:
            run.append(blk)
    if run:
        out.append(run)
    return out


def _split_table(rendered: str, table: Table, max_tokens: int = 512) -> list[str]:
    """Split a large table by row groups, repeating the header in each.

    A chunk of table body with no column names is unretrievable (the query terms
    are in the header) and uninterpretable once retrieved. Repeating the header
    costs a few tokens per chunk and is the difference between a usable unit and
    a useless one.
    """
    if estimate_tokens(rendered) <= max_tokens or not table.rows:
        return [rendered]
    lines = rendered.splitlines()
    header_lines = lines[: max(table.header_rows, 1) + 1]
    body = lines[len(header_lines) :]
    header_text = "\n".join(header_lines)
    budget = max_tokens - estimate_tokens(header_text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for line in body:
        tok = estimate_tokens(line)
        if cur and cur_tok + tok > budget:
            chunks.append(header_text + "\n" + "\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(line)
        cur_tok += tok
    if cur:
        chunks.append(header_text + "\n" + "\n".join(cur))
    return chunks or [rendered]


# Candidate boundaries; a candidate is kept only if the next character is an
# uppercase letter in any script or an opening bracket (see _is_sentence_start).
# The lookahead used to be [A-Z], so a German sentence opening with "Über" or
# an Italian one with "È" was never split.
_SENT = re.compile(r"(?<=[.!?])\s+")


def _is_sentence_start(text: str, i: int) -> bool:
    return i < len(text) and (text[i].isupper() or text[i] in "([")


def _split_long_block(text: str, max_tokens: int) -> list[tuple[int, int]]:
    """Split one oversized block at sentence boundaries.

    Returns ``(start, end)`` offsets into ``text`` rather than the pieces
    themselves, so the caller can take exact slices and give each piece a span
    that resolves. Returning strings was the earlier design and it quietly broke
    the span contract: joining sentences with a single space discards the
    original newlines, and the unit no longer matches the text it cites.

    This is the only place in the segmenter where a character count decides a
    boundary, and it is reached only when one structural block is already over
    budget on its own.
    """
    if not text:
        return [(0, len(text))]
    bounds: list[int] = [0]
    for m in _SENT.finditer(text):
        if _is_sentence_start(text, m.end()):
            bounds.append(m.end())
    bounds.append(len(text))

    out: list[tuple[int, int]] = []
    start = 0
    cur_tokens = 0
    for i in range(1, len(bounds)):
        seg_start, seg_end = bounds[i - 1], bounds[i]
        seg_tokens = estimate_tokens(text[seg_start:seg_end])
        if cur_tokens and cur_tokens + seg_tokens > max_tokens:
            out.append((start, seg_start))
            start = seg_start
            cur_tokens = 0
        cur_tokens += seg_tokens
    if start < len(text):
        out.append((start, len(text)))
    return out or [(0, len(text))]


# --------------------------------------------------------------------------- #
# fixed window -- the control arm                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FixedWindowParams:
    window_tokens: int = 512
    overlap_tokens: int = 64


@register(
    "segment",
    "fixed_window",
    version="1",
    params_model=dataclass_params(FixedWindowParams),
    summary="Fixed character windows with overlap. The control arm for structural chunking.",
)
def _make_fixed(params: dict[str, Any], **_: Any) -> FixedWindowSegmenter:
    return FixedWindowSegmenter(params)


class FixedWindowSegmenter(StageImpl):
    """Slices the canonical text into overlapping windows.

    Included so "structure beats character counts" is measured rather than
    assumed. It also demonstrates the cost the contract is avoiding: because
    windows are position-derived, inserting a paragraph early in a document
    shifts every subsequent boundary and changes every subsequent unit id --
    so an incremental rebuild re-embeds the whole document for a one-line edit.
    ``check_unit_stability`` will fail on this implementation, correctly.
    """

    STAGE, IMPL, VERSION = "segment", "fixed_window", "1"

    def segment(self, parsed: ParsedDocument, ctx: StageContext) -> Sequence[Unit]:
        window = int(self.param("window_tokens", 512)) * 4
        overlap = int(self.param("overlap_tokens", 64)) * 4
        step = max(1, window - overlap)
        b = _UnitBuilder(parsed)
        text = parsed.text
        blocks_by_span = list(parsed.blocks)

        for start in range(0, max(1, len(text)), step):
            chunk = text[start : start + window]
            if not chunk.strip():
                continue
            end = start + len(chunk)
            covering = [
                blk for blk in blocks_by_span if blk.provenance.span.overlaps(Span(start, end))
            ]
            if not covering:
                continue
            path = _path_for(parsed, start)
            h = hash_text(chunk)
            occ = b._occurrences.get(h, 0)
            b._occurrences[h] = occ + 1
            b.units.append(
                Unit(
                    unit_id=make_unit_id(parsed.document_id, h, occurrence=occ),
                    document_id=parsed.document_id,
                    text=chunk,
                    provenance=Provenance(
                        document_id=parsed.document_id,
                        span=Span(start, end),
                        source_uri=parsed.source_uri,
                        block_ids=tuple(blk.block_id for blk in covering),
                    ),
                    section_path=path,
                    ordinal=len(b.units),
                    metadata=dict(parsed.metadata),
                )
            )
        return b.finish()


def _path_for(parsed: ParsedDocument, offset: int) -> tuple[str, ...]:
    stack: list[tuple[int, str]] = []
    for blk in parsed.blocks:
        if blk.provenance.span.start > offset:
            break
        if str(blk.kind) == BlockKind.HEADING:
            level = blk.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, blk.text))
    return tuple(t for _, t in stack)


# --------------------------------------------------------------------------- #
# whole document -- the disabled-segment behaviour                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WholeDocumentParams:
    pass


@register(
    "segment",
    "whole_document",
    version="1",
    params_model=dataclass_params(WholeDocumentParams),
    summary="One unit per document. The disabled-segment arm; wins on short documents.",
)
def _make_whole(params: dict[str, Any], **_: Any) -> WholeDocumentSegmenter:
    return WholeDocumentSegmenter(params)


class WholeDocumentSegmenter(StageImpl):
    STAGE, IMPL, VERSION = "segment", "whole_document", "1"

    def segment(self, parsed: ParsedDocument, ctx: StageContext) -> Sequence[Unit]:
        if not parsed.blocks:
            return []
        b = _UnitBuilder(parsed)
        b.add(parsed.text, list(parsed.blocks), ())
        return b.finish()

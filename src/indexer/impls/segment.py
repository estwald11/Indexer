"""Segmenters: structural, items, fixed-window, and whole-document.

The contract says boundaries come from structure, not character counts. Four
implementations make that testable rather than asserted:

``structural``
    Honours the contract. Sections are the unit; oversized sections split at
    block boundaries; tables split by row groups with headers repeated.

``items``
    For documents whose structure is below the section -- specifications,
    price lists, bills of quantities, catalogues, procedures with numbered
    points: one unit per numbered entry, however short. A document with fewer
    entries than ``min_items`` is segmented as ``structural`` would, so one
    segmenter can serve an archive that mixes both.

``fixed_window``
    Deliberately violates the spirit of it. It exists because "structure-aware
    chunking beats fixed windows" is a claim, and a claim needs a control arm.

``whole_document``
    One unit per document. The disabled-segment behaviour, and a real baseline:
    on short documents it sometimes wins.

None of them makes a unit of a ``BlockKind.QUOTED`` block -- the history below
an email reply -- or lets a unit run across one. It is in the document for
enrichers to read, not to be found.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from indexer.core.document import Block, BlockKind, ParsedDocument, Table
from indexer.core.ids import hash_text, make_unit_id
from indexer.core.provenance import Provenance, Span
from indexer.core.registry import register
from indexer.core.stages import StageContext
from indexer.core.unit import Unit, UnitKind
from indexer.plugin import StageImpl, dataclass_params

__all__ = [
    "DEFAULT_CHAPTER_PATTERN",
    "DEFAULT_ITEM_PATTERN",
    "FixedWindowSegmenter",
    "ItemSegmenter",
    "StructuralSegmenter",
    "WholeDocumentSegmenter",
]


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
        """Sections into units.

        ``max_tokens`` caps a unit. ``min_tokens`` and ``overlap_tokens`` govern
        only the places where a *size* forced a boundary -- a section run past
        the cap, a single block past it -- because those are the only places
        the contract lets size decide anything: a split never leaves a piece
        under ``min_tokens``, and the piece after a split repeats up to
        ``overlap_tokens`` of the one before. Both were accepted and ignored
        before, as was ``max_tokens`` for tables, which were always cut at 512.
        """
        max_tok = int(self.param("max_tokens", 512))
        min_tok = int(self.param("min_tokens", 0))
        overlap = int(self.param("overlap_tokens", 0))
        merge_below = int(self.param("merge_below_tokens", 32))
        b = _UnitBuilder(parsed)
        quoted = [blk.provenance.span for blk in parsed.blocks if _is_quoted(blk)]

        for path, blocks in _sections(parsed):
            content = [blk for blk in blocks if str(blk.kind) != BlockKind.HEADING]
            if not content:
                continue

            for group in _split_tables(content, max_tok):
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
                # Whether the run in `pending` continues one a size limit cut.
                # Only then do min_tokens and overlap apply to it.
                continues_split = False
                carried_tokens = 0
                for blk in group:
                    tok = estimate_tokens(blk.text)
                    if tok > max_tok:
                        if pending:
                            b.add(_join(pending), pending, path)
                            pending, pending_tokens = [], 0
                        continues_split = False
                        carried_tokens = 0
                        base = blk.provenance.span.start
                        for rel_start, rel_end in _split_long_block(
                            blk.text, max_tok, min_tokens=min_tok, overlap_tokens=overlap
                        ):
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
                        # Carry the tail of the unit just closed into the next
                        # one, whole blocks only, up to the overlap budget.
                        pending = _overlap_tail(pending, overlap) if overlap else []
                        pending_tokens = sum(estimate_tokens(x.text) for x in pending)
                        carried_tokens = pending_tokens
                        continues_split = True
                    pending.append(blk)
                    pending_tokens += tok
                if pending:
                    threshold = max(merge_below, min_tok) if continues_split else merge_below
                    # Carried blocks are already in the previous unit; only what
                    # is new counts toward whether this piece stands on its own.
                    fresh = pending_tokens - carried_tokens
                    # Never folded across quoted text: the merged unit would
                    # be a slice of the document that includes it.
                    if (
                        fresh < threshold
                        and b.units
                        and not _crosses(quoted, b.units[-1].provenance.span.end, pending[0])
                    ):
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


def _is_quoted(blk: Block) -> bool:
    return str(blk.kind) == BlockKind.QUOTED


def _crosses(quoted: Sequence[Span], start: int, blk: Block) -> bool:
    """Whether quoted text lies between ``start`` and ``blk``."""
    end = blk.provenance.span.start
    return any(start <= q.start < end for q in quoted)


def _sections(parsed: ParsedDocument) -> list[tuple[tuple[str, ...], list[Block]]]:
    """Group blocks under their heading trail, preserving reading order. A
    quoted block is left out and ends the group it interrupts, so no run of
    blocks -- and no unit made of one -- spans it."""
    out: list[tuple[tuple[str, ...], list[Block]]] = []
    stack: list[tuple[int, str]] = []
    current: list[Block] = []
    path: tuple[str, ...] = ()

    for blk in parsed.blocks:
        if _is_quoted(blk):
            if current:
                out.append((path, current))
                current = []
            continue
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


def _overlap_tail(blocks: Sequence[Block], overlap_tokens: int) -> list[Block]:
    """The trailing whole blocks of a run that fit in ``overlap_tokens``."""
    tail: list[Block] = []
    used = 0
    for blk in reversed(blocks):
        tok = estimate_tokens(blk.text)
        if used + tok > overlap_tokens:
            break
        tail.insert(0, blk)
        used += tok
    # Never carry the whole run: the next unit would then begin as a copy of
    # the one before.
    return tail if len(tail) < len(blocks) else tail[1:]


def _split_tables(blocks: list[Block], max_tokens: int = 512) -> list[Any]:
    """Separate tables from prose runs so a table is never merged into a chunk."""
    out: list[Any] = []
    run: list[Block] = []
    for blk in blocks:
        if str(blk.kind) == BlockKind.TABLE and blk.table is not None:
            if run:
                out.append(run)
                run = []
            out.append((blk, _split_table(blk.text, blk.table, max_tokens)))
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


#: A sentence boundary: terminal punctuation, space, then a capital -- accented
#: ones included, or an Italian sentence opening with "È" never split.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ(\[])")


def _split_long_block(
    text: str, max_tokens: int, *, min_tokens: int = 0, overlap_tokens: int = 0
) -> list[tuple[int, int]]:
    """Split one oversized block at sentence boundaries; see ``_sentence_pieces``.

    Then, if the last piece is under ``min_tokens``, it joins the one before --
    a 30-token tail is too small to retrieve on and costs an embedding like any
    other unit. And each piece after the first starts up to ``overlap_tokens``
    earlier, at a sentence boundary, so a sentence cut from its antecedent
    carries it. Pieces are still slices of the block, so each stays verbatim.
    """
    pieces = _sentence_pieces(text, max_tokens)
    if min_tokens and len(pieces) > 1:
        last_start, last_end = pieces[-1]
        if estimate_tokens(text[last_start:last_end]) < min_tokens:
            prev_start, _ = pieces[-2]
            pieces = [*pieces[:-2], (prev_start, last_end)]
    if overlap_tokens and len(pieces) > 1:
        bounds = [0, *(m.end() for m in _SENT.finditer(text))]
        shifted = [pieces[0]]
        for start, end in pieces[1:]:
            earlier = [
                s for s in bounds if s < start and estimate_tokens(text[s:start]) <= overlap_tokens
            ]
            shifted.append((min(earlier) if earlier else start, end))
        pieces = shifted
    return pieces


def _sentence_pieces(text: str, max_tokens: int) -> list[tuple[int, int]]:
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
# items -- specifications, price lists, bills of quantities                    #
# --------------------------------------------------------------------------- #

#: An item code opening a line: three or more levels (03.02.002, 13.01.11.05*.a,
#: B.72.14.0013, 13E.201.01) or a letter and two (A.1C.4), optionally after
#: "Pos.", "Art.", "Voce" or "Nr." -- and not a date or an amount written with
#: thousands separators, which have the same shape.
DEFAULT_ITEM_PATTERN = (
    r"^[ \t]*(?:(?i:pos|art|voce|nr)\.?[ \t]*)?"
    r"(?!\d{1,2}\.\d{1,2}\.(?:19|20)\d\d\b)"
    r"(?!\d{1,3}(?:\.\d{3})+(?:,\d+)?(?![\w.]))"
    r"(?:(?:[A-Z]{1,3}\.?)?\d{1,3}[A-Z]?(?:\.\d{1,4}[A-Za-z]?){2,5}\*?(?:\.[a-z0-9]{1,2})?"
    r"|[A-Z]\.\d{1,3}[A-Z]?(?:\.\d{1,3}[A-Z]?){1,3})"
    r"[.):]?(?=[ \t]|$)"
)
#: A chapter line, for documents whose parser finds no headings (a PDF's text
#: layer): an optional code of one or two levels, then a title in capitals. The
#: title is checked beyond the pattern -- all capitals, a word of four letters
#: or more -- and a line without a code inside an item is read as part of it: a
#: brand in capitals on a line of its own does not open a chapter.
DEFAULT_CHAPTER_PATTERN = (
    r"^[ \t]*(?P<code>\d{1,3}(?:\.\d{1,3})?)?[ \t.)\-]*"
    r"(?P<title>[^\W\d_][^\n]{2,118}?)[ \t]*$"
)
#: The kind of a unit that is one item.
ITEM_KIND = "item"


@dataclass(frozen=True, slots=True)
class ItemParams:
    #: What opens an item, at the start of a line. See ``DEFAULT_ITEM_PATTERN``.
    item_pattern: str = DEFAULT_ITEM_PATTERN
    #: A document with fewer items than this is not a list of items -- a
    #: letter, a report with two numbered paragraphs -- and is segmented as
    #: ``structural`` segments it.
    min_items: int = 3
    #: ``structural``'s, for the documents segmented as it would.
    merge_below_tokens: int = 32
    #: What a chapter line looks like; the depth of its ``code`` group is its
    #: level. Empty turns chapter lines off.
    chapter_pattern: str = DEFAULT_CHAPTER_PATTERN
    max_tokens: int = 512
    min_tokens: int = 64
    overlap_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.item_pattern:
            raise ValueError("item_pattern: an item segmenter needs one")
        if self.min_items < 1:
            raise ValueError("min_items: at least 1")
        for name in ("item_pattern", "chapter_pattern"):
            pattern = getattr(self, name)
            if pattern:
                try:
                    re.compile(pattern, re.M)
                except re.error as exc:
                    raise ValueError(f"{name}: {exc}") from exc


@register(
    "segment",
    "items",
    version="2",
    params_model=dataclass_params(ItemParams),
    summary=(
        "One unit per numbered entry (specifications, price lists, bills of quantities, "
        "numbered procedures), however short; chapter lines become headings. Documents "
        "without enough entries are segmented as `structural` would."
    ),
)
def _make_items(params: dict[str, Any], **_: Any) -> ItemSegmenter:
    return ItemSegmenter(params)


class ItemSegmenter(StageImpl):
    """One unit per numbered entry of a document.

    ``structural`` makes units of sections and caps them by size. Many
    documents are structured below that: a specification's chapter is a list
    of numbered items, each something to be supplied; so are a price list, a
    bill of quantities, a catalogue, a procedure's numbered steps. A question
    is about one entry. From a PDF's text layer, which has no headings,
    ``structural`` could only fill units up to their size limit, so an item
    shared its unit with six others and a question about it was answered with
    the first 600 characters of whichever came first.

    Here an item code at the start of a line opens a unit, which runs to the
    next item, chapter line, heading or table -- across blocks, since a PDF puts
    an item's continuation on the next page in a block of its own. Items are
    never merged, however short: "Idem, diametro 26x3 mm" is an item, and giving
    it words to be found by is the reference resolver's job (``llm_resolver``).
    A size threshold that folded it into its neighbour would have decided it
    was not worth finding. An item over ``max_tokens`` splits at sentence
    boundaries, and its later pieces carry its first line as their last
    heading, so each still says what it belongs to.

    Text before a chapter's first item becomes units of its own; tables are
    split as ``structural`` splits them. A document with fewer than
    ``min_items`` items is not a list -- a letter that numbers two paragraphs
    -- and is segmented by ``structural``'s rules, which is what lets one
    configuration serve an archive of both.
    """

    STAGE, IMPL, VERSION = "segment", "items", "2"

    def segment(self, parsed: ParsedDocument, ctx: StageContext) -> Sequence[Unit]:
        items = re.compile(str(self.param("item_pattern", DEFAULT_ITEM_PATTERN)), re.M)
        if not _has_items(parsed, items, int(self.param("min_items", 3))):
            defaults = {"max_tokens": 512, "min_tokens": 0, "overlap_tokens": 0}
            params = {k: self.param(k, v) for k, v in defaults.items()}
            params["merge_below_tokens"] = self.param("merge_below_tokens", 32)
            return StructuralSegmenter(params).segment(parsed, ctx)
        pattern = str(self.param("chapter_pattern", DEFAULT_CHAPTER_PATTERN) or "")
        chapters = re.compile(pattern, re.M) if pattern else None
        max_tok = int(self.param("max_tokens", 512))
        min_tok = int(self.param("min_tokens", 0))
        overlap = int(self.param("overlap_tokens", 0))
        starts = [blk.provenance.span.start for blk in parsed.blocks]
        b = _UnitBuilder(parsed)

        for piece in _item_pieces(parsed, items, chapters):
            tblock = piece.table
            if tblock is not None and tblock.table is not None:
                chunks = _split_table(tblock.text, tblock.table, max_tok)
                split = len(chunks) > 1
                for ci, chunk_text in enumerate(chunks):
                    b.add(
                        chunk_text,
                        [tblock],
                        piece.path,
                        kind=UnitKind.TABLE_ROWS if split else UnitKind.TABLE,
                        table_ref=f"{tblock.block_id}#{ci}",
                        verbatim=not split,
                    )
                continue
            start, end = _trimmed(parsed.text, piece.start, piece.end)
            text = parsed.text[start:end]
            kind: UnitKind | str = ITEM_KIND if piece.kind == "item" else UnitKind.PROSE
            if estimate_tokens(text) <= max_tok:
                blocks = _covering(parsed, starts, start, end)
                b.add(text, blocks, piece.path, kind=kind, span=Span(start, end))
                continue
            # Its first sentence: "03.02.001 Telaio per lavabo."
            head = _SENT.split(text.split("\n", 1)[0].strip(), maxsplit=1)[0][:120]
            pieces = _split_long_block(text, max_tok, min_tokens=min_tok, overlap_tokens=overlap)
            for n, (rel_start, rel_end) in enumerate(pieces):
                path = piece.path if n == 0 or piece.kind != "item" else (*piece.path, head)
                b.add(
                    text[rel_start:rel_end],
                    _covering(parsed, starts, start + rel_start, start + rel_end),
                    path,
                    kind=kind,
                    span=Span(start + rel_start, start + rel_end),
                )
        return b.finish()


class _Piece(NamedTuple):
    start: int
    end: int
    kind: str  # item | text | table
    path: tuple[str, ...]
    table: Block | None = None


def _item_pieces(
    parsed: ParsedDocument, items: re.Pattern[str], chapters: re.Pattern[str] | None
) -> list[_Piece]:
    """The document as items, the text between them, and tables, in order."""
    pieces: list[_Piece] = []
    heads: list[tuple[int, str]] = []
    chaps: list[tuple[int, str]] = []
    cur: list[Any] = []  # start, kind and path of the piece being read

    def path() -> tuple[str, ...]:
        return tuple(t for _, t in heads) + tuple(t for _, t in chaps)

    def close(end: int) -> None:
        if cur:
            start, kind, p = cur
            if parsed.text[start:end].strip():
                pieces.append(_Piece(start, end, kind, p))
            cur.clear()

    for blk in parsed.blocks:
        span = blk.provenance.span
        if _is_quoted(blk):
            close(span.start)
            continue
        if str(blk.kind) == BlockKind.HEADING:
            close(span.start)
            level = blk.level or 1
            while heads and heads[-1][0] >= level:
                heads.pop()
            heads.append((level, blk.text))
            chaps.clear()
            continue
        if str(blk.kind) == BlockKind.TABLE and blk.table is not None:
            close(span.start)
            pieces.append(_Piece(span.start, span.end, "table", path(), blk))
            continue
        marks: dict[int, tuple[str, Any]] = {}
        if chapters is not None:
            for m in chapters.finditer(blk.text):
                found = _chapter(m)
                if found is not None:
                    marks[m.start()] = ("chapter", (m.end(), *found))
        # A line that opens an item is an item, whatever else it looks like.
        for m in items.finditer(blk.text):
            marks[m.start()] = ("item", None)
        if not cur:
            cur.extend((span.start, "text", path()))
        for rel in sorted(marks):
            what, info = marks[rel]
            if what == "item":
                close(span.start + rel)
                cur.extend((span.start + rel, "item", path()))
                continue
            line_end, level, title, coded = info
            if not coded and cur and cur[1] == "item":
                continue
            close(span.start + rel)
            while chaps and chaps[-1][0] >= level:
                chaps.pop()
            chaps.append((level, title))
            cur.extend((span.start + line_end, "text", path()))
    close(len(parsed.text))
    return pieces


def _has_items(parsed: ParsedDocument, items: re.Pattern[str], at_least: int) -> bool:
    """Whether ``parsed`` opens at least ``at_least`` items."""
    found = 0
    for blk in parsed.blocks:
        if str(blk.kind) in (BlockKind.HEADING, BlockKind.TABLE, BlockKind.QUOTED):
            continue
        for _ in items.finditer(blk.text):
            found += 1
            if found >= at_least:
                return True
    return False


_TITLE_WORD = re.compile(r"[^\W\d_]{4,}")


def _chapter(m: re.Match[str]) -> tuple[int, str, bool] | None:
    """A chapter line's level, text and whether it has a code -- or None when
    the line only matches the pattern: a title must be in capitals."""
    groups = m.groupdict()
    title = (groups.get("title") or m.group(0)).strip()
    letters = [c for c in title if c.isalpha()]
    if len(letters) < 6 or any(c.islower() for c in letters) or not _TITLE_WORD.search(title):
        return None
    code = groups.get("code") or ""
    return (code.count(".") + 1 if code else 1), m.group(0).strip(), bool(code)


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _covering(parsed: ParsedDocument, starts: Sequence[int], start: int, end: int) -> list[Block]:
    """The blocks a span of the canonical text overlaps, in reading order."""
    first = max(0, bisect_left(starts, start + 1) - 1)
    out: list[Block] = []
    for blk in parsed.blocks[first:]:
        if blk.provenance.span.start >= end:
            break
        if blk.provenance.span.end > start:
            out.append(blk)
    return out or [parsed.blocks[min(first, len(parsed.blocks) - 1)]]


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
        blocks_by_span = [blk for blk in parsed.blocks if not _is_quoted(blk)]

        for start, end in _windows(parsed, window, step):
            chunk = text[start:end]
            if not chunk.strip():
                continue
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


def _regions(parsed: ParsedDocument) -> list[tuple[int, int]]:
    """The stretches of the text between quoted blocks: all of it, if none."""
    quoted = [blk.provenance.span for blk in parsed.blocks if _is_quoted(blk)]
    if not quoted:
        return [(0, len(parsed.text))]
    out: list[tuple[int, int]] = []
    start: int | None = None
    end = 0
    for blk in parsed.blocks:
        if _is_quoted(blk):
            if start is not None:
                out.append((start, end))
            start = None
            continue
        if start is None:
            start = blk.provenance.span.start
        end = blk.provenance.span.end
    if start is not None:
        out.append((start, end))
    return out


def _windows(parsed: ParsedDocument, window: int, step: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in _regions(parsed):
        for start in range(lo, max(lo + 1, hi), step):
            out.append((start, min(start + window, hi)))
    return out


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
        if not any(_is_quoted(blk) for blk in parsed.blocks):
            b.add(parsed.text, list(parsed.blocks), ())
            return b.finish()
        # One unit per stretch of the document's own text.
        starts = [blk.provenance.span.start for blk in parsed.blocks]
        for start, end in _regions(parsed):
            b.add(
                parsed.text[start:end],
                _covering(parsed, starts, start, end),
                (),
                span=Span(start, end),
            )
        return b.finish()

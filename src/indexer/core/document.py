"""Source documents and the output of `parse`: ordered typed blocks.

The parse contract is "ordered typed blocks with provenance; never loses reading
order or table structure". Three design choices carry that:

1.  ``ParsedDocument.blocks`` *is* the reading order. There is no separate order
    field to disagree with the list, and no implementation is permitted to
    return blocks in raster or stream order while claiming otherwise --
    ``reading_order_confidence`` exists so a parser that cannot recover order
    declares it instead of degrading silently.

2.  A table is a ``Table`` payload, not a string. ``Block.text`` for a table is a
    *rendering* (markdown, by convention) for the benefit of text indexes; the
    grid remains addressable for the structured index and for a segmenter that
    must not split a row. Downstream code that needs cells and reads ``text``
    is doing it wrong, and the type makes that visible.

3.  ``ParsedDocument.text`` is the canonical linearisation, and block spans index
    into it. This is what makes provenance checkable rather than asserted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from indexer.core.ids import ContentHash, DocumentId, hash_text
from indexer.core.provenance import Provenance

__all__ = [
    "LOCATOR_KEYS",
    "Block",
    "BlockKind",
    "MediaRef",
    "ParsedDocument",
    "SourceDocument",
    "Table",
    "TableCell",
    "content_metadata",
]

#: Metadata keys that say where the bytes live *on this machine* and nothing
#: about what they are. Excluded from every cache key and content hash: an
#: absolute path in a key makes the cache machine-specific, so moving an archive
#: to a new server would re-pay every enrichment -- the LLM calls included --
#: while appearing to work. Scanners keep them for ``load()``; nothing else
#: should read them.
LOCATOR_KEYS = frozenset({"path"})


def content_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Metadata minus locators: the part that can legitimately change output.

    Keys starting with an underscore are a scanner's private bookkeeping (how to
    reach an attachment inside an email, say) and are excluded for the same
    reason as ``path``.
    """
    return {k: v for k, v in metadata.items() if k not in LOCATOR_KEYS and not k.startswith("_")}


class BlockKind(StrEnum):
    """What a block *is*, as far as the parser can tell.

    Open by convention: an implementation may emit a kind not listed here and
    the frame will carry it through. Nothing in the frame branches on kind
    except segmenters, which are allowed to and must tolerate unknown values.
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    CODE = "code"
    FORMULA = "formula"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    TOC_ENTRY = "toc_entry"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class TableCell:
    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


@dataclass(frozen=True, slots=True)
class Table:
    """A table's structure, preserved independently of its rendering.

    ``rows`` is row-major and ragged-tolerant: implementations that recover a
    partial grid emit what they have rather than padding with lies.
    """

    rows: tuple[tuple[TableCell, ...], ...]
    caption: str | None = None
    #: Number of leading rows that are headers. Segmenters repeat these when a
    #: large table is split, so each unit stays independently interpretable.
    header_rows: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), max((len(r) for r in self.rows), default=0))


@dataclass(frozen=True, slots=True)
class MediaRef:
    """A pointer to extracted binary media (a page raster, a figure crop).

    Held by reference, never inline: a visual index needs the pixels, everything
    else needs to not pay for them. ``uri`` is resolved by the artifact store.
    """

    uri: str
    media_type: str
    content_hash: ContentHash
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class Block:
    """One typed, positioned piece of a document.

    ``text`` is always populated, including for tables and figures (a rendering
    and a caption/alt-text respectively), so that a text-only consumer never has
    to special-case. Structure lives alongside, never instead.
    """

    block_id: str
    kind: BlockKind | str
    text: str
    provenance: Provenance
    #: Heading depth, 1-based. Meaningful for ``HEADING``; ``None`` elsewhere.
    level: int | None = None
    table: Table | None = None
    media: MediaRef | None = None
    #: Parser-specific extras (font size, confidence, language). Carried, never
    #: interpreted by the frame; anything the frame needs has a typed home.
    attrs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> ContentHash:
        return hash_text(self.text)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A document as found by a corpus scanner, before parsing.

    ``content`` is bytes rather than a path so that a scanner may serve from
    object storage or a database without every parser learning how. Scanners
    that want laziness supply a loader and leave ``content`` empty until read.
    """

    document_id: DocumentId
    source_uri: str
    content_hash: ContentHash
    media_type: str
    size_bytes: int
    #: Caller metadata that survives to every hit: tenant, matter, effective
    #: date. Distinct from fields *extracted* by enrich, which are inferred.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def load(self) -> bytes:
        """Fetch the bytes. Overridden by scanners; the base holds none."""
        raise NotImplementedError("SourceDocument.load is supplied by the corpus scanner")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """The output of `parse`. Blocks in reading order plus their linearisation.

    Invariant, asserted by ``indexer.eval.checks``:
        ``text[b.provenance.span.start : b.provenance.span.end] == b.text``
        for every block, and block spans are non-overlapping and ascending.
    """

    document_id: DocumentId
    source_uri: str
    #: Canonical linearisation. All spans in the pipeline index into this.
    text: str
    blocks: tuple[Block, ...]
    #: Hash of the *source* bytes this was parsed from. The cache key for every
    #: downstream stage descends from it.
    source_hash: ContentHash
    page_count: int | None = None
    #: 0.0-1.0. Below 1.0 means the parser could not fully determine reading
    #: order (multi-column, no layout model) and downstream quality will suffer.
    #: Recorded in the manifest so a bad corpus is visible without a bisect.
    reading_order_confidence: float = 1.0
    #: Per-page rasters, populated only when a visual index is configured.
    page_media: tuple[MediaRef, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> ContentHash:
        return hash_text(self.text)

    def blocks_by_kind(self, *kinds: BlockKind | str) -> Sequence[Block]:
        wanted = {str(k) for k in kinds}
        return [b for b in self.blocks if str(b.kind) in wanted]

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]

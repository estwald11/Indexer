"""Provenance: the chain from a returned passage back to ink on a page.

The engineering property is "any returned passage traces to document, page,
span". That is only enforceable if every stage is required to *carry* a span
forward rather than reconstruct one at the end, and if "span" means something
checkable.

Span semantics
--------------
A ``Span`` is a half-open character range into the **canonical text** of a
parsed document (``ParsedDocument.text``) -- not into the original bytes. For a
PDF, offsets into the original bytes are meaningless; for HTML they point at
markup rather than content. The canonical text is the parser's own linearised
rendering, and the frame requires it to satisfy:

    parsed.text[block.span.start : block.span.end] == block.text

That equality is machine-checkable (``indexer.eval`` asserts it), which turns
provenance from a promise into a test. Where a byte-level anchor into the
original source *is* meaningful, parsers additionally populate ``source_span``.

``PageRef`` and ``BBox`` are optional because not every format has pages or
geometry, and a contract that demands them would force text parsers to invent
values. Anything that has them MUST carry them: they are what a citation UI and
a visual index both need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, TypeVar

from indexer.core.ids import DocumentId

__all__ = ["BBox", "PageRef", "Provenance", "Span"]


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open ``[start, end)`` character range into a canonical text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def overlap_length(self, other: Span) -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def contains(self, other: Span) -> bool:
        return self.start <= other.start and other.end <= self.end

    def union(self, other: Span) -> Span:
        return Span(min(self.start, other.start), max(self.end, other.end))


@dataclass(frozen=True, slots=True)
class BBox:
    """Geometry on a page, in PDF points, origin top-left.

    Present only where the parser recovers layout. The visual index and any
    citation overlay depend on it; text-only parsers leave it ``None``.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )


@dataclass(frozen=True, slots=True)
class PageRef:
    """A page range, 1-based and inclusive. A single page has start == end."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ValueError(f"invalid page range {self.start}..{self.end}")

    @classmethod
    def of(cls, page: int) -> Self:
        return cls(page, page)

    def union(self, other: PageRef) -> PageRef:
        return PageRef(min(self.start, other.start), max(self.end, other.end))


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a piece of content came from. Carried by every block, unit and hit.

    ``document_id`` and ``span`` are mandatory: they are the minimum needed to
    re-derive the exact text from the parsed document. Everything else is
    present when the source format supports it.
    """

    document_id: DocumentId
    span: Span
    source_uri: str = ""
    pages: PageRef | None = None
    bbox: BBox | None = None
    #: Offsets into the *original* bytes, when the format makes that meaningful
    #: (markdown, html, plain text). ``None`` for PDF, DOCX and scans.
    source_span: Span | None = None
    #: Ids of the parsed blocks this content was derived from, in reading order.
    block_ids: tuple[str, ...] = field(default_factory=tuple)

    def merged_with(self, other: Provenance) -> Provenance:
        """Combine provenance of adjacent content (segmenting blocks into a unit).

        Only valid within one document; merging across documents is a bug that
        would produce an uncitable passage, so it raises rather than picking one.
        """
        if self.document_id != other.document_id:
            raise ValueError(
                f"cannot merge provenance across documents: "
                f"{self.document_id} != {other.document_id}"
            )
        return Provenance(
            document_id=self.document_id,
            span=self.span.union(other.span),
            source_uri=self.source_uri or other.source_uri,
            pages=_union_opt(self.pages, other.pages),
            bbox=_union_opt(self.bbox, other.bbox),
            source_span=_union_opt(self.source_span, other.source_span),
            block_ids=self.block_ids + other.block_ids,
        )


_U = TypeVar("_U", PageRef, BBox, Span)


def _union_opt(a: _U | None, b: _U | None) -> _U | None:
    if a is None:
        return b
    if b is None:
        return a
    return a.union(b)

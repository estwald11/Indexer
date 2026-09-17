"""Parsers: text, markdown, reStructuredText, and a routing parser.

All four maintain the parse contract's span invariant by construction: each
builds ``ParsedDocument.text`` by *appending* block text and recording the
offsets as it goes, rather than parsing first and locating spans afterwards.
Locating afterwards is where span bugs come from -- the same string appears
twice and ``str.find`` returns the first one.

Tables are real ``Table`` payloads, not rendered strings. A markdown pipe table
that arrives as text has already destroyed the structured path; recovering it
downstream is guesswork.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from indexer.core.document import Block, BlockKind, ParsedDocument, SourceDocument, Table, TableCell
from indexer.core.provenance import Provenance, Span
from indexer.core.registry import register
from indexer.core.stages import StageContext
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["MarkdownParser", "PassthroughParser", "RoutingParser", "RstParser", "TextParser"]


class _Builder:
    """Accumulates blocks and the canonical text together.

    The contract is ``text[span] == block.text``. Building both in one pass is
    the only way to make that true by construction rather than by inspection.
    """

    __slots__ = ("_n", "blocks", "document_id", "offset", "parts", "source_uri")

    SEP = "\n\n"

    def __init__(self, document_id: str, source_uri: str) -> None:
        self.parts: list[str] = []
        self.blocks: list[Block] = []
        self.offset = 0
        self.document_id = document_id
        self.source_uri = source_uri
        self._n = 0

    def add(
        self,
        text: str,
        kind: BlockKind | str,
        *,
        level: int | None = None,
        table: Table | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Block:
        if self.parts:
            self.parts.append(self.SEP)
            self.offset += len(self.SEP)
        start = self.offset
        self.parts.append(text)
        self.offset += len(text)
        block = Block(
            block_id=f"b{self._n}",
            kind=kind,
            text=text,
            provenance=Provenance(
                document_id=self.document_id,  # type: ignore[arg-type]
                span=Span(start, self.offset),
                source_uri=self.source_uri,
            ),
            level=level,
            table=table,
            attrs=attrs or {},
        )
        self._n += 1
        self.blocks.append(block)
        return block

    def finish(self, source_hash: Any, **kw: Any) -> ParsedDocument:
        return ParsedDocument(
            document_id=self.document_id,  # type: ignore[arg-type]
            source_uri=self.source_uri,
            text="".join(self.parts),
            blocks=tuple(self.blocks),
            source_hash=source_hash,
            **kw,
        )


def _decode(doc: SourceDocument, encoding: str) -> str:
    raw = doc.load()
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        # Replace rather than fail: one bad byte in a 400-page document should
        # not remove the document from the corpus. The substitution is visible
        # in the text, so it is diagnosable.
        return raw.decode(encoding, errors="replace")


# --------------------------------------------------------------------------- #
# text                                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TextParams:
    encoding: str = "utf-8"


@register(
    "parse",
    "text",
    version="1",
    params_model=dataclass_params(TextParams),
    summary="Split on blank lines. The minimal implementation of the parse contract.",
)
def _make_text(params: dict[str, Any], **_: Any) -> TextParser:
    return TextParser(params)


class TextParser(StageImpl):
    """The minimal parse implementation: paragraphs on blank lines.

    Claims full reading-order confidence because for plain text the file order
    *is* the reading order -- there is nothing to recover and nothing to get
    wrong.
    """

    STAGE, IMPL, VERSION = "parse", "text", "1"

    def can_parse(self, doc: SourceDocument) -> float:
        return 0.3 if doc.media_type.startswith("text/") else 0.1

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        text = _decode(doc, self.param("encoding", "utf-8"))
        b = _Builder(doc.document_id, doc.source_uri)
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                b.add(para, BlockKind.PARAGRAPH)
        return b.finish(doc.content_hash, metadata=dict(doc.metadata))


# --------------------------------------------------------------------------- #
# markdown                                                                     #
# --------------------------------------------------------------------------- #

_H = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^```|^~~~")
_LIST = re.compile(r"^\s*([-*+]|\d+\.)\s+")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


@dataclass(frozen=True, slots=True)
class MarkdownParams:
    encoding: str = "utf-8"
    keep_code: bool = True


@register(
    "parse",
    "markdown",
    version="1",
    params_model=dataclass_params(MarkdownParams),
    summary="Headings, paragraphs, lists, fenced code and pipe tables with structure preserved.",
)
def _make_markdown(params: dict[str, Any], **_: Any) -> MarkdownParser:
    return MarkdownParser(params)


class MarkdownParser(StageImpl):
    STAGE, IMPL, VERSION = "parse", "markdown", "1"

    def can_parse(self, doc: SourceDocument) -> float:
        return 0.9 if doc.media_type == "text/markdown" else 0.2

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        text = _decode(doc, self.param("encoding", "utf-8"))
        b = _Builder(doc.document_id, doc.source_uri)
        lines = text.splitlines()
        i, n = 0, len(lines)
        buf: list[str] = []

        def flush(kind: BlockKind = BlockKind.PARAGRAPH) -> None:
            nonlocal buf
            body = "\n".join(buf).strip()
            if body:
                b.add(body, kind)
            buf = []

        while i < n:
            line = lines[i]

            if _FENCE.match(line.strip()):
                flush()
                fence = line.strip()[:3]
                code = [line]
                i += 1
                while i < n and not lines[i].strip().startswith(fence):
                    code.append(lines[i])
                    i += 1
                if i < n:
                    code.append(lines[i])
                    i += 1
                if self.param("keep_code", True):
                    b.add("\n".join(code), BlockKind.CODE)
                continue

            m = _H.match(line)
            if m:
                flush()
                b.add(m.group(2).strip(), BlockKind.HEADING, level=len(m.group(1)))
                i += 1
                continue

            # A pipe table is detected by its separator row, then consumed whole.
            # The grid is preserved; the rendering in `text` is a convenience.
            if "|" in line and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
                flush()
                rows_raw = [line, lines[i + 1]]
                i += 2
                while i < n and "|" in lines[i] and lines[i].strip():
                    rows_raw.append(lines[i])
                    i += 1
                b.add(
                    "\n".join(rows_raw),
                    BlockKind.TABLE,
                    table=_parse_pipe_table(rows_raw),
                )
                continue

            if _LIST.match(line):
                flush()
                items: list[str] = []
                while i < n and (_LIST.match(lines[i]) or (lines[i].startswith("  ") and items)):
                    items.append(lines[i])
                    i += 1
                b.add("\n".join(items).strip(), BlockKind.LIST_ITEM)
                continue

            if not line.strip():
                flush()
                i += 1
                continue

            buf.append(line)
            i += 1

        flush()
        return b.finish(doc.content_hash, metadata=dict(doc.metadata))


def _parse_pipe_table(rows_raw: list[str]) -> Table:
    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    header = cells(rows_raw[0])
    body = [cells(r) for r in rows_raw[2:]]
    return Table(
        rows=(
            tuple(TableCell(c, is_header=True) for c in header),
            *(tuple(TableCell(c) for c in r) for r in body),
        ),
        header_rows=1,
    )


# --------------------------------------------------------------------------- #
# reStructuredText                                                             #
# --------------------------------------------------------------------------- #

_RST_UNDERLINE = re.compile(r"^([=\-`:'\"~^_*+#])\1{1,}\s*$")
_RST_DIRECTIVE = re.compile(r"^\.\.\s+(\w[\w-]*)::")
#: Underline characters in the order Python's own docs use them. rst has no
#: fixed heading levels -- the order of first appearance defines them -- but
#: assuming the conventional order recovers the right depth for the large
#: majority of real documents, and getting it wrong only costs section-path
#: precision rather than correctness.
_RST_LEVELS = "#*=-^\"'`~:+_"


@dataclass(frozen=True, slots=True)
class RstParams:
    encoding: str = "utf-8"


@register(
    "parse",
    "rst",
    version="1",
    params_model=dataclass_params(RstParams),
    summary="reStructuredText: underline headings, directives, literal blocks, simple tables.",
)
def _make_rst(params: dict[str, Any], **_: Any) -> RstParser:
    return RstParser(params)


class RstParser(StageImpl):
    STAGE, IMPL, VERSION = "parse", "rst", "1"

    def can_parse(self, doc: SourceDocument) -> float:
        return 0.9 if doc.media_type == "text/x-rst" else 0.15

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        text = _decode(doc, self.param("encoding", "utf-8"))
        b = _Builder(doc.document_id, doc.source_uri)
        lines = text.splitlines()
        i, n = 0, len(lines)
        buf: list[str] = []
        seen_levels: list[str] = []

        def flush() -> None:
            nonlocal buf
            body = "\n".join(buf).strip()
            if body:
                b.add(body, BlockKind.PARAGRAPH)
            buf = []

        while i < n:
            line = lines[i]

            if (
                i + 1 < n
                and line.strip()
                and _RST_UNDERLINE.match(lines[i + 1] or "")
                and len(lines[i + 1].strip()) >= len(line.strip()) - 1
            ):
                flush()
                ch = lines[i + 1].strip()[0]
                if ch not in seen_levels:
                    seen_levels.append(ch)
                b.add(line.strip(), BlockKind.HEADING, level=seen_levels.index(ch) + 1)
                i += 2
                continue

            m = _RST_DIRECTIVE.match(line)
            if m:
                flush()
                directive_lines = [line]
                i += 1
                while i < n and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                    directive_lines.append(lines[i])
                    i += 1
                kind = (
                    BlockKind.CODE
                    if m.group(1) in ("code", "code-block", "sourcecode")
                    else BlockKind.OTHER
                )
                b.add("\n".join(directive_lines).rstrip(), kind, attrs={"directive": m.group(1)})
                continue

            if line.rstrip().endswith("::"):
                buf.append(line)
                flush()
                i += 1
                literal: list[str] = []
                while i < n and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                    literal.append(lines[i])
                    i += 1
                literal_text = "\n".join(literal).strip("\n")
                if literal_text.strip():
                    b.add(literal_text, BlockKind.CODE)
                continue

            if not line.strip():
                flush()
                i += 1
                continue

            buf.append(line)
            i += 1

        flush()
        return b.finish(doc.content_hash, metadata=dict(doc.metadata))


# --------------------------------------------------------------------------- #
# passthrough (the disabled-parse behaviour)                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PassthroughParams:
    encoding: str = "utf-8"


@register(
    "parse",
    "passthrough",
    version="1",
    params_model=dataclass_params(PassthroughParams),
    summary="Whole document as one block. The disabled-parse arm; a real ablation baseline.",
)
def _make_passthrough(params: dict[str, Any], **_: Any) -> PassthroughParser:
    return PassthroughParser(params)


class PassthroughParser(StageImpl):
    STAGE, IMPL, VERSION = "parse", "passthrough", "1"

    def can_parse(self, doc: SourceDocument) -> float:
        return 0.05

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        text = _decode(doc, self.param("encoding", "utf-8")).strip()
        b = _Builder(doc.document_id, doc.source_uri)
        if text:
            b.add(text, BlockKind.OTHER)
        # Honest about what it did not do: no structure was recovered, so
        # downstream quality is expected to suffer and the manifest will say so.
        return b.finish(doc.content_hash, reading_order_confidence=1.0, metadata=dict(doc.metadata))


# --------------------------------------------------------------------------- #
# routing                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RoutingParams:
    routes: list[dict[str, Any]] = field(default_factory=list)
    default: dict[str, Any] = field(default_factory=lambda: {"impl": "text"})


@register(
    "parse",
    "routing",
    version="1",
    params_model=dataclass_params(RoutingParams),
    summary="Dispatches per document by media type and metadata. First match wins.",
)
def _make_routing(params: dict[str, Any], **_: Any) -> RoutingParser:
    return RoutingParser(params)


class RoutingParser(StageImpl):
    """Per-document parser selection.

    A corpus is not one format. Choosing a parser globally means either paying
    scan prices for text-native documents or losing structure on the scans, and
    both are avoidable at the cost of a match expression.

    Note the fingerprint: it covers the *routing table plus every routed
    parser's fingerprint*, so changing a downstream parser's version correctly
    invalidates the documents that route to it -- and only those.
    """

    STAGE, IMPL, VERSION = "parse", "routing", "1"

    def __init__(self, params: dict[str, Any], resolver: Any = None) -> None:
        super().__init__(params)
        from indexer.config.loader import resolve_impl

        self._resolve = resolver or (lambda spec: resolve_impl("parse", spec))
        self._routes: list[tuple[dict[str, Any], Any]] = []
        for route in params.get("routes", []):
            when = {k: v for k, v in route.items() if k not in ("impl", "params")}
            reg, norm = self._resolve({"impl": route["impl"], "params": route.get("params", {})})
            self._routes.append((when.get("when", when), reg.build(norm)))
        d = params.get("default", {"impl": "text"})
        reg, norm = self._resolve({"impl": d["impl"], "params": d.get("params", {})})
        self._default = reg.build(norm)

    def fingerprint(self) -> Any:
        from indexer.core.accounting import StageFingerprint
        from indexer.core.ids import hash_obj

        return StageFingerprint(
            stage=self.STAGE,
            impl=self.IMPL,
            version=self.VERSION,
            params_hash=hash_obj(
                {
                    "params": self._params,
                    "routes": [p.fingerprint().key() for _, p in self._routes],
                    "default": self._default.fingerprint().key(),
                }
            ),
        )

    def _pick(self, doc: SourceDocument) -> Any:
        for when, parser in self._routes:
            if _matches(when, doc):
                return parser
        return self._default

    def can_parse(self, doc: SourceDocument) -> float:
        return float(self._pick(doc).can_parse(doc))

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        parsed: ParsedDocument = self._pick(doc).parse(doc, ctx)
        return parsed


def _matches(when: dict[str, Any], doc: SourceDocument) -> bool:
    for key, want in when.items():
        if key == "media_type":
            if doc.media_type != want:
                return False
        elif key == "extension":
            if not str(doc.metadata.get("name", "")).endswith(str(want)):
                return False
        elif key == "max_bytes":
            if doc.size_bytes > int(want):
                return False
        elif key == "min_bytes":
            if doc.size_bytes < int(want):
                return False
        elif doc.metadata.get(key) != want:
            return False
    return True

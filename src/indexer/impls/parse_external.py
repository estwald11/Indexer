"""Layout-aware and managed parsers: Docling, pymupdf4llm, LlamaParse.

The reference parsers read text. Scans, multi-column layouts, forms and complex
tables need a layout model, and ``configs/full.yaml`` has named three for a
long time without any existing. These are adapters: each maps its engine's
output onto ``ParsedDocument`` -- headings with levels, tables as grids, page
numbers on every block -- and imports the engine only when it parses, so
registering them costs nothing.

``docling``       Self-hosted, MIT. Layout analysis, table structure, OCR for
                  scans. The default choice for an archive: nothing leaves the
                  premises.
``pymupdf4llm``   Fast markdown from text-layer PDFs. **AGPL-3.0** (PyMuPDF):
                  fine inside a company, a licensing question inside a product.
``llamaparse``    A managed API. Strong on hard tables, and the document is
                  **uploaded to a third party**: under GDPR that needs a legal
                  basis and a processing agreement, and the EU endpoint keeps
                  the data in the EU. The API key is read from config, which
                  takes it from the environment.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from indexer.core.document import BlockKind, ParsedDocument, SourceDocument
from indexer.core.provenance import BBox, PageRef
from indexer.core.registry import register
from indexer.core.stages import StageContext
from indexer.impls.parse import _Builder, markdown_into
from indexer.impls.parse_office import render_table
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["DoclingParser", "LlamaParseParser", "PyMuPDF4LLMParser", "docling_to_blocks"]

_PDF_LIKE = frozenset(
    {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
)


# --------------------------------------------------------------------------- #
# docling                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DoclingParams:
    ocr: bool = True
    #: OCR languages, for engines that take them.
    ocr_languages: list[str] = field(default_factory=lambda: ["it", "en"])
    table_structure: bool = True
    #: Running headers and footers repeat on every page and say nothing about
    #: the content; the segment contract allows leaving them out.
    keep_page_furniture: bool = False


@register(
    "parse",
    "docling",
    version="1",
    params_model=dataclass_params(DoclingParams),
    summary="Docling (MIT, self-hosted): layout, reading order, table structure, OCR for scans.",
    requires=("docling",),
)
def _make_docling(params: dict[str, Any], **kw: Any) -> DoclingParser:
    return DoclingParser(params, convert=kw.get("convert"))


class DoclingParser(StageImpl):
    """Docling behind the parse contract.

    ``convert`` -- bytes and a file name in, a ``DoclingDocument`` out -- is
    injectable, which is how the mapping is tested without the models; by
    default it is built lazily, once, because constructing a converter loads
    them.
    """

    STAGE, IMPL, VERSION = "parse", "docling", "1"

    def __init__(
        self, params: dict[str, Any], convert: Callable[[bytes, str], Any] | None = None
    ) -> None:
        super().__init__(params)
        self._convert = convert

    def can_parse(self, doc: SourceDocument) -> float:
        if doc.media_type in _PDF_LIKE:
            return 0.9
        if doc.media_type.endswith(("wordprocessingml.document", "html")):
            return 0.6
        return 0.0

    def _converter(self) -> Callable[[bytes, str], Any]:
        if self._convert is not None:
            return self._convert
        try:
            from docling.datamodel.base_models import DocumentStream, InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("the docling parser needs docling: pip install docling") from exc

        options = PdfPipelineOptions()
        options.do_ocr = bool(self.param("ocr", True))
        options.do_table_structure = bool(self.param("table_structure", True))
        langs = self.param("ocr_languages", ["it", "en"])
        if langs and hasattr(options, "ocr_options") and hasattr(options.ocr_options, "lang"):
            options.ocr_options.lang = list(langs)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

        def convert(data: bytes, name: str) -> Any:
            return converter.convert(DocumentStream(name=name, stream=io.BytesIO(data))).document

        self._convert = convert
        return convert

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        name = str(doc.metadata.get("name") or "document.pdf")
        dl_doc = self._converter()(doc.load(), name)
        b = _Builder(doc.document_id, doc.source_uri)
        docling_to_blocks(dl_doc, b, keep_furniture=bool(self.param("keep_page_furniture", False)))
        pages = getattr(dl_doc, "pages", None) or {}
        return b.finish(
            doc.content_hash,
            page_count=len(pages) or None,
            metadata=dict(doc.metadata),
        )


def _label(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def docling_to_blocks(dl_doc: Any, b: _Builder, *, keep_furniture: bool = False) -> None:
    """Map a ``DoclingDocument`` onto blocks, in its reading order.

    Duck-typed on purpose -- labels, ``prov``, ``level``, ``data.grid`` -- so a
    minor docling-core change does not turn into a parse failure. Section
    levels are shifted below the title when there is one, so the title stays
    the root of every section path rather than being popped by the first
    level-1 section.
    """
    items = [item for item, _ in dl_doc.iterate_items()]
    has_title = any(_label(i) == "title" for i in items)
    for item in items:
        label = _label(item)
        pages, bbox = _where(dl_doc, item)
        text = str(getattr(item, "text", "") or "").strip()
        if label in ("page_header", "page_footer") and not keep_furniture:
            continue
        if label == "table":
            rows, header = _grid(item)
            if rows:
                rendered, table = render_table(rows, header_rows=header)
                b.add(rendered, BlockKind.TABLE, table=table, pages=pages, bbox=bbox)
            continue
        if label in ("picture", "chart"):
            caption = _caption(item, dl_doc)
            if caption:
                b.add(caption, BlockKind.FIGURE, pages=pages, bbox=bbox)
            continue
        if not text:
            continue
        if label == "title":
            b.add(text, BlockKind.HEADING, level=1, pages=pages, bbox=bbox)
        elif label == "section_header":
            level = int(getattr(item, "level", 1) or 1) + (1 if has_title else 0)
            b.add(text, BlockKind.HEADING, level=min(level, 6), pages=pages, bbox=bbox)
        else:
            kind = {
                "list_item": BlockKind.LIST_ITEM,
                "code": BlockKind.CODE,
                "formula": BlockKind.FORMULA,
                "caption": BlockKind.CAPTION,
                "footnote": BlockKind.FOOTNOTE,
                "page_header": BlockKind.PAGE_HEADER,
                "page_footer": BlockKind.PAGE_FOOTER,
            }.get(label, BlockKind.PARAGRAPH)
            b.add(text, kind, pages=pages, bbox=bbox)


def _where(dl_doc: Any, item: Any) -> tuple[PageRef | None, BBox | None]:
    provs = getattr(item, "prov", None) or []
    if not provs:
        return None, None
    prov = provs[0]
    page_no = getattr(prov, "page_no", None)
    pages = PageRef.of(int(page_no)) if page_no else None
    box = getattr(prov, "bbox", None)
    bbox = None
    if box is not None:
        height = None
        page = (getattr(dl_doc, "pages", None) or {}).get(page_no)
        if page is not None and getattr(page, "size", None) is not None:
            height = page.size.height
        if height is not None and hasattr(box, "to_top_left_origin"):
            box = box.to_top_left_origin(page_height=height)
        try:
            left, top, right, bottom = float(box.l), float(box.t), float(box.r), float(box.b)
            bbox = BBox(min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        except (AttributeError, TypeError, ValueError):
            bbox = None
    return pages, bbox


def _grid(item: Any) -> tuple[list[list[str]], int]:
    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None) or []
    rows = [[str(getattr(c, "text", "") or "") for c in row] for row in grid]
    header = 0
    for row in grid:
        if row and all(getattr(c, "column_header", False) for c in row):
            header += 1
        else:
            break
    return [r for r in rows if any(x.strip() for x in r)], header


def _caption(item: Any, dl_doc: Any) -> str:
    fn = getattr(item, "caption_text", None)
    if callable(fn):
        try:
            return str(fn(dl_doc) or "").strip()
        except TypeError:
            return str(fn() or "").strip()
    return ""


# --------------------------------------------------------------------------- #
# pymupdf4llm                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PyMuPDF4LLMParams:
    keep_code: bool = True


@register(
    "parse",
    "pymupdf4llm",
    version="1",
    params_model=dataclass_params(PyMuPDF4LLMParams),
    summary=(
        "Fast markdown from text-layer PDFs, page by page (pymupdf4llm). AGPL-3.0: "
        "check it fits your distribution."
    ),
    requires=("pymupdf4llm",),
)
def _make_pymupdf4llm(params: dict[str, Any], **kw: Any) -> PyMuPDF4LLMParser:
    return PyMuPDF4LLMParser(params, to_markdown=kw.get("to_markdown"))


class PyMuPDF4LLMParser(StageImpl):
    STAGE, IMPL, VERSION = "parse", "pymupdf4llm", "1"

    def __init__(
        self,
        params: dict[str, Any],
        to_markdown: Callable[[bytes], list[dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__(params)
        self._to_markdown = to_markdown

    def can_parse(self, doc: SourceDocument) -> float:
        return 0.85 if doc.media_type == "application/pdf" else 0.0

    def _pages(self, data: bytes) -> list[dict[str, Any]]:
        if self._to_markdown is not None:
            return self._to_markdown(data)
        try:
            import pymupdf
            import pymupdf4llm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "the pymupdf4llm parser needs pymupdf4llm: pip install pymupdf4llm"
            ) from exc
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            return list(pymupdf4llm.to_markdown(pdf, page_chunks=True))

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        chunks = self._pages(doc.load())
        b = _Builder(doc.document_id, doc.source_uri)
        for n, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata") or {}
            # "page_number" in current releases, "page" in older ones; both 1-based.
            page = int(meta.get("page_number") or meta.get("page") or n)
            markdown_into(
                b,
                str(chunk.get("text") or ""),
                keep_code=bool(self.param("keep_code", True)),
                pages=PageRef.of(page),
            )
        return b.finish(doc.content_hash, page_count=len(chunks), metadata=dict(doc.metadata))


# --------------------------------------------------------------------------- #
# llamaparse                                                                   #
# --------------------------------------------------------------------------- #

#: (method, url, headers, body) -> (status, body). Injectable for tests and for
#: deployments that must route through a proxy with its own client.
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]


def _urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return int(resp.status), bytes(resp.read())
    except urllib.error.HTTPError as err:
        return int(err.code), bytes(err.read() or b"")


@dataclass(frozen=True, slots=True)
class LlamaParseParams:
    api_key: str = ""
    #: https://api.cloud.eu.llamaindex.ai keeps documents in the EU region.
    base_url: str = "https://api.cloud.eu.llamaindex.ai"
    tier: str = "cost_effective"  # fast | cost_effective | agentic | agentic_plus
    version: str = "latest"
    #: Further keys for the job's configuration JSON, passed through.
    configuration: dict[str, Any] = field(default_factory=dict)
    poll_interval_s: float = 2.0
    timeout_s: float = 600.0


@register(
    "parse",
    "llamaparse",
    version="1",
    params_model=dataclass_params(LlamaParseParams),
    summary=(
        "LlamaParse managed API (v2), markdown per page. Uploads the document to a "
        "third party: needs a GDPR basis; defaults to the EU region."
    ),
)
def _make_llamaparse(params: dict[str, Any], **kw: Any) -> LlamaParseParser:
    return LlamaParseParser(params, transport=kw.get("transport"), sleep=kw.get("sleep"))


class LlamaParseParser(StageImpl):
    """Upload, poll, fetch markdown per page, map pages to blocks.

    The fingerprint covers the tier, version and configuration -- the output
    depends on all three -- and not the key, which the loader redacts: rotating
    a key must not re-parse an archive.
    """

    STAGE, IMPL, VERSION = "parse", "llamaparse", "1"

    def __init__(
        self,
        params: dict[str, Any],
        transport: Transport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__({k: v for k, v in params.items()})
        self._transport = transport or _urllib_transport
        self._sleep = sleep or time.sleep

    def fingerprint(self) -> Any:
        from indexer.core.accounting import StageFingerprint
        from indexer.core.ids import hash_obj

        return StageFingerprint(
            stage=self.STAGE,
            impl=self.IMPL,
            version=self.VERSION,
            params_hash=hash_obj({k: v for k, v in self._params.items() if k != "api_key"}),
        )

    def can_parse(self, doc: SourceDocument) -> float:
        return 0.8 if doc.media_type in _PDF_LIKE else 0.0

    def _call(self, method: str, path: str, body: bytes | None, ctype: str | None) -> Any:
        key = str(self.param("api_key", ""))
        if not key:
            raise RuntimeError("llamaparse needs api_key (set it from ${env:LLAMA_CLOUD_API_KEY})")
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        if ctype:
            headers["Content-Type"] = ctype
        url = str(self.param("base_url", "https://api.cloud.eu.llamaindex.ai")).rstrip("/") + path
        status, raw = self._transport(method, url, headers, body)
        if status >= 400:
            raise RuntimeError(f"llamaparse {method} {path}: HTTP {status}: {raw[:300]!r}")
        return json.loads(raw.decode("utf-8") or "{}")

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        name = str(doc.metadata.get("name") or "document.pdf")
        configuration = {
            "tier": self.param("tier", "cost_effective"),
            "version": self.param("version", "latest"),
            **dict(self.param("configuration", {}) or {}),
        }
        body, ctype = _multipart(
            {"configuration": json.dumps(configuration)}, "file", name, doc.load()
        )
        job = self._call("POST", "/api/v2/parse/upload", body, ctype)
        job_id = job.get("id") or (job.get("job") or {}).get("id")
        if not job_id:
            raise RuntimeError(f"llamaparse upload returned no job id: {job}")
        deadline = time.monotonic() + float(self.param("timeout_s", 600.0))
        while True:
            state = self._call("GET", f"/api/v2/parse/{job_id}", None, None)
            status = str((state.get("job") or state).get("status", "")).upper()
            if status == "COMPLETED":
                break
            if status in ("FAILED", "CANCELLED"):
                err = (state.get("job") or {}).get("error_message", "")
                raise RuntimeError(f"llamaparse job {job_id} {status.lower()}: {err}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"llamaparse job {job_id} did not finish in time")
            self._sleep(float(self.param("poll_interval_s", 2.0)))
        result = self._call("GET", f"/api/v2/parse/{job_id}?expand=markdown", None, None)
        pages = (result.get("markdown") or {}).get("pages") or []
        b = _Builder(doc.document_id, doc.source_uri)
        for n, page in enumerate(pages, start=1):
            number = int(page.get("page_number") or n)
            markdown_into(b, str(page.get("markdown") or ""), pages=PageRef.of(number))
        meta = dict(doc.metadata)
        meta["parsed_by"] = "llamaparse"
        return b.finish(doc.content_hash, page_count=len(pages) or None, metadata=meta)


def _multipart(
    fields: dict[str, str], file_field: str, filename: str, data: bytes
) -> tuple[bytes, str]:
    boundary = f"indexer-{uuid.uuid4().hex}"
    out = io.BytesIO()
    for key, value in fields.items():
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        out.write(value.encode("utf-8") + b"\r\n")
    safe = filename.replace('"', "")
    out.write(f"--{boundary}\r\n".encode())
    out.write(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{safe}"\r\n'.encode()
    )
    out.write(b"Content-Type: application/octet-stream\r\n\r\n")
    out.write(data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"

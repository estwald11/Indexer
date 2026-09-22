"""Docling, pymupdf4llm and LlamaParse adapters, against stand-ins for the engines.

The engines are heavy (models) or remote (an API), so the tests drive each
adapter's *mapping* with objects shaped like the engine's output -- the part
this repo owns and can get wrong -- and run the parse contract check on every
result.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.document import SourceDocument
from indexer.core.ids import DocumentId, hash_bytes
from indexer.core.stages import StageContext
from indexer.eval.checks import check_parsed_document
from indexer.impls.parse_external import DoclingParser, LlamaParseParser, PyMuPDF4LLMParser

CTX = StageContext(cache=NullCache(), accountant=InMemoryAccountant())


class _Bytes(SourceDocument):
    __slots__ = ()

    def load(self) -> bytes:
        return self.metadata["_bytes"]  # type: ignore[no-any-return]


def _doc(data: bytes = b"%PDF-1.4 fake") -> SourceDocument:
    return _Bytes(
        document_id=DocumentId("d"),
        source_uri="mem://scan.pdf",
        content_hash=hash_bytes(data),
        media_type="application/pdf",
        size_bytes=len(data),
        metadata={"_bytes": data, "name": "scan.pdf"},
    )


class _Box:
    def __init__(self, l: float, t: float, r: float, b: float) -> None:  # noqa: E741
        self.l, self.t, self.r, self.b = l, t, r, b

    def to_top_left_origin(self, page_height: float) -> _Box:
        return _Box(self.l, page_height - self.t, self.r, page_height - self.b)


def _item(label: str, text: str = "", page: int = 1, **kw: Any) -> NS:
    return NS(
        label=NS(value=label),
        text=text,
        prov=[NS(page_no=page, bbox=_Box(72, 700, 300, 680))],
        **kw,
    )


def _cell(text: str, header: bool = False) -> NS:
    return NS(text=text, column_header=header)


class _DoclingDoc:
    def __init__(self, items: list[NS]) -> None:
        self._items = items
        self.pages = {1: NS(size=NS(width=612, height=792)), 2: NS(size=NS(width=612, height=792))}

    def iterate_items(self) -> Any:
        return ((i, 0) for i in self._items)


class TestDocling:
    def test_layout_items_map_onto_blocks_with_pages_and_boxes(self) -> None:
        table = _item(
            "table",
            page=2,
            data=NS(
                grid=[
                    [_cell("Voce", True), _cell("Importo", True)],
                    [_cell("Canone"), _cell("1.000,00")],
                ]
            ),
        )
        picture = _item("picture", page=2, caption_text=lambda doc: "Figura 1: organigramma")
        dl = _DoclingDoc(
            [
                _item("page_header", "Azienda Spa - riservato"),
                _item("title", "Relazione annuale 2024"),
                _item("section_header", "Risultati", level=1),
                _item("text", "Il fatturato è cresciuto del 12%."),
                _item("list_item", "Nuovi clienti: 40"),
                table,
                picture,
                _item("page_footer", "Pagina 1"),
            ]
        )
        parser = DoclingParser({}, convert=lambda data, name: dl)
        parsed = parser.parse(_doc(), CTX)
        assert check_parsed_document(parsed) == []
        kinds = [(str(b.kind), b.level) for b in parsed.blocks]
        assert kinds == [
            ("heading", 1),
            ("heading", 2),  # below the title, so the title stays the root
            ("paragraph", None),
            ("list_item", None),
            ("table", None),
            ("figure", None),
        ]
        assert not any("riservato" in b.text or "Pagina 1" in b.text for b in parsed.blocks)
        tbl = parsed.blocks[4]
        assert tbl.table is not None and tbl.table.header_rows == 1
        assert tbl.provenance.pages is not None and tbl.provenance.pages.start == 2
        box = parsed.blocks[0].provenance.bbox
        assert box is not None and box.y0 < box.y1 and box.y0 == 792 - 700
        assert parsed.page_count == 2


class TestPyMuPDF4LLM:
    def test_markdown_pages_keep_their_page_numbers(self) -> None:
        chunks = [
            {"metadata": {"page_number": 1}, "text": "# Contratto\n\nArticolo 1. Oggetto."},
            {
                "metadata": {"page": 2},  # an older release's key
                "text": "| Voce | Importo |\n|---|---|\n| Canone | 1.000 |",
            },
        ]
        parser = PyMuPDF4LLMParser({}, to_markdown=lambda data: chunks)
        parsed = parser.parse(_doc(), CTX)
        assert check_parsed_document(parsed) == []
        assert [(str(b.kind), b.provenance.pages.start) for b in parsed.blocks] == [  # type: ignore[union-attr]
            ("heading", 1),
            ("paragraph", 1),
            ("table", 2),
        ]


class _FakeLlamaParse:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.polls = 0

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        self.calls.append((method, url, headers, body))
        if url.endswith("/api/v2/parse/upload"):
            return 200, json.dumps({"id": "job-1", "status": "PENDING"}).encode()
        if url.endswith("/api/v2/parse/job-1"):
            self.polls += 1
            status = "COMPLETED" if self.polls > 1 else "RUNNING"
            return 200, json.dumps({"job": {"id": "job-1", "status": status}}).encode()
        if url.endswith("expand=markdown"):
            pages = [
                {"page_number": 1, "markdown": "# Verbale\n\nPresenti: 5 soci."},
                {"page_number": 2, "markdown": "Deliberazioni approvate."},
            ]
            return 200, json.dumps(
                {"job": {"status": "COMPLETED"}, "markdown": {"pages": pages}}
            ).encode()
        return 404, b"not found"


class TestLlamaParse:
    def test_upload_poll_and_markdown_per_page(self) -> None:
        fake = _FakeLlamaParse()
        parser = LlamaParseParser(
            {"api_key": "secret", "tier": "agentic"}, transport=fake, sleep=lambda s: None
        )
        parsed = parser.parse(_doc(b"%PDF-1.4 verbale"), CTX)
        assert check_parsed_document(parsed) == []
        assert [(b.text, b.provenance.pages.start) for b in parsed.blocks] == [  # type: ignore[union-attr]
            ("Verbale", 1),
            ("Presenti: 5 soci.", 1),
            ("Deliberazioni approvate.", 2),
        ]
        method, url, headers, body = fake.calls[0]
        assert (method, url) == ("POST", "https://api.cloud.eu.llamaindex.ai/api/v2/parse/upload")
        assert headers["Authorization"] == "Bearer secret"
        assert body is not None and b'"tier": "agentic"' in body and b"%PDF-1.4 verbale" in body
        assert fake.polls == 2

    def test_the_key_is_not_part_of_the_fingerprint(self) -> None:
        a = LlamaParseParser({"api_key": "old"})
        b = LlamaParseParser({"api_key": "rotated"})
        c = LlamaParseParser({"api_key": "old", "tier": "agentic"})
        assert a.fingerprint() == b.fingerprint()
        assert a.fingerprint() != c.fingerprint()

    def test_a_failed_job_is_an_error(self) -> None:
        def failing(
            method: str, url: str, headers: dict[str, str], body: bytes | None
        ) -> tuple[int, bytes]:
            if url.endswith("upload"):
                return 200, b'{"id": "j"}'
            return 200, b'{"job": {"status": "FAILED", "error_message": "unsupported file"}}'

        parser = LlamaParseParser({"api_key": "k"}, transport=failing, sleep=lambda s: None)
        try:
            parser.parse(_doc(), CTX)
        except RuntimeError as exc:
            assert "unsupported file" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("a failed job must raise")

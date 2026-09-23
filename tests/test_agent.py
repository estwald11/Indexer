"""The agent's tool set: what each tool returns, and what it must not."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from indexer.agent import UNTRUSTED, AgentTools, ToolError
from indexer.cli import main
from indexer.mcp_server import INSTRUCTIONS, build_server
from indexer.pipeline import assemble

CONFIG = """
schema_version: 1
project: {{name: agent}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params:
        root: {data}
        include: ["**/*.md"]
        acl_rules:
          - {{pattern: "amministrazione/**", acl: ["group:amm"]}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural, max_tokens: 60, params: {{merge_below_tokens: 0}}}}
  enrich:
    enabled: true
    enrichers:
      - {{impl: section_prefix, scope: unit}}
      - {{impl: entities}}
      - impl: regex_fields
        params:
          locale: it
          fields:
            importo: {{pattern: 'Totale: € ([0-9.,]+)', type: float}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25, params: {{fallback_language: it}}}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: rules
    params: {{level: document}}
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
  access: {{enabled: {access}, missing: allow}}
"""

FILES = {
    "amministrazione/fattura-1.md": (
        "# Fattura 1\n\nFornitore Rossi S.r.l., P.IVA 01234567897.\n\n"
        "## Importi\n\nTotale: € 1.250,00\n\n"
        "## Pagamento\n\nBonifico su IBAN IT60 X054 2811 1010 0000 0123 456 entro 30 giorni.\n"
    ),
    "amministrazione/fattura-2.md": (
        "# Fattura 2\n\nFornitore Bianchi S.p.A., P.IVA 12345678903.\n\n"
        "## Importi\n\nTotale: € 480,00\n\n"
        "## Note\n\nIl codice P.IVA 01234567890 riportato nell'ordine è errato.\n"
    ),
    "pubblico/procedura.md": (
        "# Procedura acquisti\n\nOgni ordine sopra 1.000 euro richiede due firme.\n\n"
        "## Fornitori\n\nIl fornitore Rossi S.r.l. (P.IVA 01234567897) è qualificato.\n"
    ),
}


def _setup(tmp_path: Path, *, access: bool = False) -> tuple[Any, Path]:
    data = tmp_path / "data"
    for rel, body in FILES.items():
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        CONFIG.format(root=tmp_path.as_posix(), data=data.as_posix(), access=str(access).lower()),
        encoding="utf-8",
    )
    a = assemble(cfg)
    assert a.ingestion().build().ok
    return a, cfg


def _doc(a: Any, name: str) -> str:
    return next(str(r.document_id) for r in a.ledger.iter_records() if r.source_uri.endswith(name))


@pytest.fixture
def archive(tmp_path: Path) -> Any:
    return _setup(tmp_path)[0]


class TestSearch:
    def test_passages_come_with_citations_and_the_build_they_reflect(self, archive: Any) -> None:
        out = AgentTools(archive).search("bonifico IBAN pagamento", top_k=2)
        assert out["as_of"]["build_id"] and out["untrusted_text"] == UNTRUSTED
        top = out["results"][0]
        assert "Bonifico" in top["text"]
        assert top["source"] == "amministrazione/fattura-1.md"
        assert top["section"] == "Fattura 1 > Pagamento"
        assert set(top["citation"]) == {"document_id", "unit_id", "source_uri", "span"}

    def test_the_cursor_pages_through_results(self, archive: Any) -> None:
        tools = AgentTools(archive)
        first = tools.search("fornitore Rossi", top_k=1)
        assert first["next_cursor"] == "1"
        second = tools.search("fornitore Rossi", top_k=1, cursor=first["next_cursor"])
        assert second["results"][0]["unit_id"] != first["results"][0]["unit_id"]

    def test_a_question_for_a_number_gets_rows(self, archive: Any) -> None:
        out = AgentTools(archive).search("fatture con importo superiore a 1000")
        assert out["route"]["path"] == "structured"
        assert [r["importo"] for r in out["rows"]] == [1250.0]
        assert out["total"] == 1

    def test_filters_narrow_the_search(self, archive: Any) -> None:
        out = AgentTools(archive).search(
            "fornitore", filters=[{"field": "importo", "op": "lt", "value": "1.000,00"}]
        )
        assert {r["source"] for r in out["results"]} == {"amministrazione/fattura-2.md"}


class TestRecords:
    def test_filters_and_aggregates(self, archive: Any) -> None:
        tools = AgentTools(archive)
        rows = tools.query_records(filters=[{"field": "importo", "op": "gt", "value": 1000}])
        assert rows["columns"] == ["document_id", "importo"] and rows["total"] == 1
        total = tools.query_records(aggregate={"op": "sum", "field": "importo"})
        assert total["rows"] == [{"sum_importo": 1730.0}]

    def test_an_unknown_field_names_the_known_ones(self, archive: Any) -> None:
        with pytest.raises(ToolError, match=r"unknown field 'imporot'.*importo"):
            AgentTools(archive).query_records(filters={"imporot": 5})

    def test_schema_describes_types_and_ranges(self, archive: Any) -> None:
        schema = AgentTools(archive).describe_schema()
        importo = next(f for f in schema["fields"] if f["name"] == "importo")
        assert importo["type"] == "float" and (importo["min"], importo["max"]) == (480.0, 1250.0)
        assert "gte" in schema["filter"]["ops"]


class TestDocuments:
    def test_a_document_is_read_a_page_at_a_time(self, archive: Any) -> None:
        tools = AgentTools(archive)
        doc = _doc(archive, "fattura-1.md")
        first = tools.get_document(doc, max_chars=80)
        assert first["document"]["title"] == "Fattura 1"
        assert first["document"]["fields"]["importo"] == 1250.0
        assert len(first["text"]) == 80 and first["next_offset"] == 80
        rest = tools.get_document(doc, offset=first["next_offset"], max_chars=10_000)
        whole = first["text"] + rest["text"]
        assert "# Fattura 1 > Importi" in whole and rest["next_offset"] is None

    def test_outline_and_expand(self, archive: Any) -> None:
        tools = AgentTools(archive)
        doc = _doc(archive, "fattura-1.md")
        sections = tools.outline(doc)["sections"]
        assert [s["section"] for s in sections] == [
            "Fattura 1",
            "Fattura 1 > Importi",
            "Fattura 1 > Pagamento",
        ]
        around = tools.expand(sections[1]["unit_ids"][0])
        assert "Rossi" in around["before"][0]["text"]
        assert "Bonifico" in around["after"][0]["text"]

    def test_an_identifier_finds_every_document_naming_it(self, archive: Any) -> None:
        tools = AgentTools(archive)
        found = tools.find_entity("IT 01234567897")
        assert found["kind"] == "piva"
        assert sorted(r["relpath"] for r in found["rows"]) == [
            "amministrazione/fattura-1.md",
            "pubblico/procedura.md",
        ]
        iban = tools.find_entity("IT60X0542811101000000123456")
        assert [r["relpath"] for r in iban["rows"]] == ["amministrazione/fattura-1.md"]


class TestAccess:
    def test_every_tool_honours_the_caller(self, tmp_path: Path) -> None:
        a, _ = _setup(tmp_path, access=True)
        outsider = AgentTools(a, principals=("group:sales",))
        insider = AgentTools(a, principals=("group:amm",))
        doc = _doc(a, "fattura-1.md")

        assert {r["source"] for r in outsider.search("fornitore Rossi")["results"]} == {
            "pubblico/procedura.md"
        }
        with pytest.raises(ToolError, match="no document"):
            outsider.get_document(doc)
        assert insider.get_document(doc)["document"]["title"] == "Fattura 1"
        assert [r["relpath"] for r in outsider.find_entity("01234567897")["rows"]] == [
            "pubblico/procedura.md"
        ]
        assert outsider.query_records(aggregate={"op": "count"})["rows"] == [{"count": 1}]
        # Values seen across documents would disclose what the caller cannot read.
        importo = next(f for f in outsider.describe_schema()["fields"] if f["name"] == "importo")
        assert "max" not in importo and "examples" not in importo


class _FakeServer:
    def __init__(self, name: str, instructions: str = "") -> None:
        self.name, self.instructions = name, instructions
        self.tools: dict[str, Any] = {}

    def tool(self) -> Any:
        def register(fn: Any) -> Any:
            self.tools[fn.__name__] = fn
            return fn

        return register


def test_the_mcp_server_exposes_every_tool(archive: Any) -> None:
    server = build_server(AgentTools(archive), server_factory=_FakeServer)
    assert server.instructions == INSTRUCTIONS
    assert sorted(server.tools) == [
        "describe_schema",
        "expand",
        "find_entity",
        "get_document",
        "outline",
        "query_records",
        "search",
    ]
    assert server.tools["search"]("bonifico")["results"]
    assert all(fn.__doc__ for fn in server.tools.values())
    with pytest.raises(ToolError):
        server.tools["get_document"]("not-a-document")


def test_the_cli_builds_answers_and_lists_the_review_queue(tmp_path: Path) -> None:
    _, cfg = _setup(tmp_path)
    buf = io.StringIO()
    assert main(["build", str(cfg), "--quiet"], out=buf) == 0
    assert json.loads(buf.getvalue())["documents"]["total"] == 3

    buf = io.StringIO()
    assert main(["query", str(cfg), "bonifico IBAN"], out=buf) == 0
    assert json.loads(buf.getvalue())["results"][0]["source"] == "amministrazione/fattura-1.md"

    buf = io.StringIO()
    assert main(["review", str(cfg)], out=buf) == 0
    items = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert [(i["enricher"], i["field"], i["value"]) for i in items] == [
        ("entities", "piva", "IT01234567890")
    ]

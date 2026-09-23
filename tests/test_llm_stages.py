"""The model-backed stages inside the pipeline: what they send, what they keep,
what a failure does, and what a batch prefill saves. A fake client answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from indexer.core.errors import ConfigError
from indexer.core.ledger import ChangeKind
from indexer.core.query import RoutePath
from indexer.pipeline import assemble
from indexer.pipeline.build import declared_fields
from indexer.pipeline.ingest import INCOMPLETE
from llm_fakes import FakeClient, message, prompt_of, schema_of

CONFIG = """
schema_version: 1
project: {{name: llm}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params: {{root: {data}, include: ["**/*.md"]}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural, max_tokens: 60, params: {{merge_below_tokens: 0}}}}
  enrich:
    enabled: true
    batch_size: {batch_size}
    max_concurrency: 3
    on_error: {on_error}
    enrichers:
{enrichers}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: {router}
    params: {router_params}
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
"""

CONTEXT = "      - {impl: llm_contextualizer, params: {mode: auto}}"
CLASSIFY = (
    "      - impl: llm_classifier\n"
    "        params: {labels: {doc_type: [fattura, contratto, altro]}}"
)
EXTRACT = """      - impl: llm_field_extractor
        params:
          schemas:
            fattura:
              importo_totale: {type: float, description: Totale documento}
              data_scadenza: {type: date}
              numero: {type: str}
              piva_fornitore: {type: str, validate: piva}
            contratto:
              controparte: {type: str}
              data_stipula: {type: date}"""

INVOICE = """# Fattura n. 42/2024

Fornitore Rossi S.r.l., P.IVA 01234567897.

## Importi

Totale documento: € 1.250,00

## Pagamento

Pagamento a 30 giorni data fattura.
"""
CONTRACT = """# Contratto di fornitura

Tra Alfa S.p.A. e Beta S.r.l., stipulato il 15/03/2024.

## Durata

Il contratto dura due anni dalla stipula.
"""


def _setup(
    tmp_path: Path,
    respond: Any,
    *,
    enrichers: str = CONTEXT,
    on_error: str = "skip",
    batch_size: int = 16,
    router: str = "rules",
    router_params: str = "{}",
    files: dict[str, str] | None = None,
) -> tuple[Any, FakeClient]:
    data = tmp_path / "data"
    for rel, body in (files or {"fattura.md": INVOICE}).items():
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        CONFIG.format(
            root=tmp_path.as_posix(),
            data=data.as_posix(),
            enrichers=enrichers,
            on_error=on_error,
            batch_size=batch_size,
            router=router,
            router_params=router_params,
        ),
        encoding="utf-8",
    )
    client = FakeClient(respond)
    return assemble(cfg, llm_client=client), client


def _units(a: Any) -> list[Any]:
    store = a.unit_store
    return [store.get(uid) for uid in sorted(store._units)]


def _contexts(params: dict[str, Any]) -> Any:
    """Answer a contextualiser request: grouped or per unit."""
    ids = list(schema_of(params).get("properties", {}))
    if ids:
        return message(data={cid: f"Contesto {cid} della fattura 42" for cid in ids})
    return message("Contesto della fattura 42 di Rossi")


def _classify(params: dict[str, Any]) -> Any:
    text = prompt_of(params).casefold()
    kind = "fattura" if "fattura n." in text else "contratto" if "contratto" in text else "altro"
    return message(data={"doc_type": kind}, model=params["model"])


def _extract(params: dict[str, Any]) -> Any:
    props = schema_of(params)["properties"]
    if "importo_totale" in props:
        data = {
            "importo_totale": {"value": 1250.0, "evidence": "Totale documento: € 1.250,00"},
            # Computed, not stated: the quote is real but names no date.
            "data_scadenza": {"value": "2024-05-30", "evidence": "Pagamento a 30 giorni"},
            # Invented: the quote is not in the document.
            "numero": {"value": "43/2024", "evidence": "Fattura n. 43/2024"},
            "piva_fornitore": {"value": "01234567897", "evidence": "P.IVA 01234567897"},
        }
    else:
        data = {
            "controparte": {"value": "Beta S.r.l.", "evidence": "Beta S.r.l."},
            "data_stipula": {"value": "2024-03-15", "evidence": "stipulato il 15/03/2024"},
        }
    return message(data=data, model=params["model"])


def _all(params: dict[str, Any]) -> Any:
    props = schema_of(params).get("properties", {})
    if "doc_type" in props:
        return _classify(params)
    if "importo_totale" in props or "controparte" in props:
        return _extract(params)
    return _contexts(params)


# --------------------------------------------------------------------------- #
# contextualiser                                                               #
# --------------------------------------------------------------------------- #


class TestContextualiser:
    def test_a_short_document_is_contextualised_in_one_call(self, tmp_path: Path) -> None:
        """Under Haiku's 4,096-token cache minimum a per-unit call re-sends the
        whole document every time; one grouped call sends it once."""
        a, client = _setup(tmp_path, _contexts)
        result = a.ingestion().build()
        assert result.ok
        units = _units(a)
        assert len(units) == 3 and len(client.calls) == 1
        call = client.calls[0]
        assert list(schema_of(call)["properties"]) == ["c1", "c2", "c3"]
        assert "language the document is written in" in call["system"]
        assert all(u.indexing_text().startswith("Contesto c") for u in units)
        enrich = result.manifest.stage_totals["enrich"]
        assert enrich["cost_usd"] > 0 and enrich["tokens_in"] == 1000

    def test_a_long_document_is_read_once_through_the_cache(self, tmp_path: Path) -> None:
        enrichers = "      - {impl: llm_contextualizer, params: {group_below_tokens: 10}}"
        a, client = _setup(tmp_path, _contexts, enrichers=enrichers)
        assert a.ingestion().build().ok
        assert len(client.calls) == 3
        for call in client.calls:
            document = call["messages"][0]["content"][0]
            assert document["cache_control"] == {"type": "ephemeral"}
            assert "Fattura n. 42/2024" in document["text"]

    def test_batch_size_bounds_the_units_per_call(self, tmp_path: Path) -> None:
        a, client = _setup(tmp_path, _contexts, batch_size=2)
        assert a.ingestion().build().ok
        # Two units in one grouped call; the one left over needs no JSON map.
        assert [len(schema_of(c).get("properties", {})) for c in client.calls] == [2, 0]

    def test_an_invalid_mode_is_a_config_error(self, tmp_path: Path) -> None:
        enrichers = "      - {impl: llm_contextualizer, params: {mode: sometimes}}"
        a, _ = _setup(tmp_path, _contexts, enrichers=enrichers)
        with pytest.raises(ConfigError, match="mode"):
            a.ingestion()


class TestFailures:
    @staticmethod
    def _refuse_once(state: dict[str, int]) -> Any:
        def respond(params: dict[str, Any]) -> Any:
            if "Pagamento a 30 giorni" in prompt_of(params).split("<chunk>")[-1]:
                state["refusals"] += 1
                if state["refusals"] == 1:
                    return message("", stop="refusal")
            return message("Contesto")

        return respond

    def test_a_refused_unit_is_indexed_and_retried_by_the_next_build(self, tmp_path: Path) -> None:
        enrichers = "      - {impl: llm_contextualizer, params: {mode: per_unit}}"
        state = {"refusals": 0}
        a, client = _setup(tmp_path, self._refuse_once(state), enrichers=enrichers)

        first = a.ingestion().build()
        assert first.ok and first.manifest.corpus.enrichments_failed == 1
        units = _units(a)
        assert len(units) == 3
        assert sum(1 for u in units if "llm_contextualizer" in u.enrichments) == 2
        (record,) = [a.ledger.get(d) for d in a.ledger.document_ids()]
        assert record.stage_keys["enrich:llm_contextualizer"] == INCOMPLETE

        calls = len(client.calls)
        plan = a.ingestion().plan()
        assert [c.kind for c in plan] == [ChangeKind.RESTAGED]
        assert plan[0].reason.startswith("retrying incomplete enrichment")
        second = a.ingestion().build()
        assert second.manifest.corpus.enrichments_failed == 0
        assert len(client.calls) - calls == 1  # only the refused unit is asked again
        assert all("llm_contextualizer" in u.enrichments for u in _units(a))
        assert [c.kind for c in a.ingestion().plan()] == [ChangeKind.UNCHANGED]

    def test_on_error_fail_fails_the_document(self, tmp_path: Path) -> None:
        enrichers = "      - {impl: llm_contextualizer, params: {mode: per_unit}}"
        a, _ = _setup(
            tmp_path, self._refuse_once({"refusals": 0}), enrichers=enrichers, on_error="fail"
        )
        result = a.ingestion().build()
        assert not result.ok and result.failures[0].stage == "enrich:llm_contextualizer"

    def test_a_bad_key_stops_the_build(self, tmp_path: Path) -> None:
        class AuthError(Exception):
            status_code = 401

        a, client = _setup(tmp_path, lambda p: AuthError("invalid x-api-key"))
        with pytest.raises(Exception, match="invalid x-api-key"):
            a.ingestion().build()
        assert len(client.calls) == 1  # not once per document of the corpus


# --------------------------------------------------------------------------- #
# classifier and field extractor                                               #
# --------------------------------------------------------------------------- #


class TestClassifierAndExtractor:
    def test_one_call_classifies_every_unit_of_a_document(self, tmp_path: Path) -> None:
        a, client = _setup(tmp_path, _classify, enrichers=CLASSIFY)
        assert a.ingestion().build().ok
        assert len(client.calls) == 1
        call = client.calls[0]
        assert schema_of(call)["properties"]["doc_type"]["enum"] == [
            "fattura",
            "contratto",
            "altro",
        ]
        assert "name: fattura.md" in prompt_of(call)
        units = _units(a)
        assert len(units) == 3
        assert {u.enrichments["llm_classifier"].labels["doc_type"] for u in units} == {"fattura"}
        assert all(u.fields()["doc_type"] == "fattura" for u in units)

    def test_the_router_knows_the_classifier_s_field(self, tmp_path: Path) -> None:
        a, _ = _setup(tmp_path, _all, enrichers=f"{CLASSIFY}\n{EXTRACT}")
        names, types = declared_fields(a.config)
        assert {"doc_type", "importo_totale", "data_stipula"} <= set(names)
        assert types["importo_totale"] == "float" and types["data_stipula"] == "date"

    def test_the_extractor_keeps_only_what_its_evidence_states(self, tmp_path: Path) -> None:
        a, client = _setup(tmp_path, _all, enrichers=f"{CLASSIFY}\n{EXTRACT}")
        assert a.ingestion().build().ok
        assert len(client.calls) == 2  # one classification, one extraction
        unit = _units(a)[0]
        e = unit.enrichments["llm_field_extractor"]
        # Stored as FatturaPA writes it, so the two sources join.
        assert dict(e.fields) == {"importo_totale": 1250.0, "piva_fornitore": "IT01234567897"}
        rejected = {r["field"]: r["reason"] for r in e.extra["rejected"]}
        assert rejected == {
            "data_scadenza": "the quoted evidence does not state this value",
            "numero": "the quoted evidence is not in the document",
        }
        records = a.query_engine().query("fatture con importo_totale superiore a 1000").records
        assert records is not None and not records.is_empty()

    def test_each_document_type_gets_its_own_schema(self, tmp_path: Path) -> None:
        a, client = _setup(
            tmp_path,
            _all,
            enrichers=f"{CLASSIFY}\n{EXTRACT}",
            files={"fattura.md": INVOICE, "contratto.md": CONTRACT},
        )
        assert a.ingestion().build().ok
        extractions = [c for c in client.calls if "doc_type" not in schema_of(c)["properties"]]
        assert sorted(sorted(schema_of(c)["properties"]) for c in extractions) == [
            ["controparte", "data_stipula"],
            ["data_scadenza", "importo_totale", "numero", "piva_fornitore"],
        ]
        contract = next(u for u in _units(a) if u.unit.metadata["name"] == "contratto.md")
        assert str(contract.fields()["data_stipula"]) == "2024-03-15"

    def test_a_reclassified_document_is_re_extracted(self, tmp_path: Path) -> None:
        a, client = _setup(tmp_path, _all, enrichers=f"{CLASSIFY}\n{EXTRACT}")
        assert a.ingestion().build().ok
        before = len(client.calls)
        # The same text, now classified by a model that says "altro": the
        # extractor's key includes the type, so it runs again, with no schema.
        client.respond = lambda p: (
            message(data={"doc_type": "altro"})
            if "doc_type" in schema_of(p)["properties"]
            else _all(p)
        )
        cfg = tmp_path / "c.yaml"
        cfg.write_text(cfg.read_text().replace("altro]", "altro, verbale]"), encoding="utf-8")
        b = assemble(cfg, llm_client=client)
        assert b.ingestion().build().ok
        assert len(client.calls) == before + 1  # classified again; nothing to extract
        assert "importo_totale" not in _units(b)[0].fields()


# --------------------------------------------------------------------------- #
# batch prefill                                                                #
# --------------------------------------------------------------------------- #


def test_prefill_answers_a_whole_build_through_batches(tmp_path: Path) -> None:
    a, client = _setup(
        tmp_path,
        _all,
        enrichers=f"{CONTEXT}\n{CLASSIFY}\n{EXTRACT}",
        files={"fattura.md": INVOICE, "contratto.md": CONTRACT},
    )
    report = a.ingestion().prefill(sleep=lambda s: None)
    # Contexts and classes first; extraction needs the class, so it waits a round.
    assert report.rounds == 3
    # One answer per distinct cache key: five units to contextualise, but one
    # classification and one extraction per document.
    assert report.by_enricher == {
        "llm_contextualizer": 5,
        "llm_classifier": 2,
        "llm_field_extractor": 2,
    }
    assert report.failed == 0 and report.cost_usd > 0
    assert not client.calls

    result = a.ingestion().build()
    assert result.ok
    assert not client.calls  # the build found every answer cached
    assert result.manifest.stage_totals["enrich"]["cost_usd"] == 0
    assert any(u.fields().get("importo_totale") == 1250.0 for u in _units(a))


def test_prefill_leaves_failed_answers_to_the_build(tmp_path: Path) -> None:
    def respond(params: dict[str, Any]) -> Any:
        if "doc_type" in schema_of(params).get("properties", {}):
            return RuntimeError("overloaded")
        return _all(params)

    a, client = _setup(tmp_path, respond, enrichers=f"{CLASSIFY}\n{EXTRACT}")
    report = a.ingestion().prefill(sleep=lambda s: None)
    assert report.rounds == 1 and report.failed == 1 and report.cached == 0
    client.respond = _all
    assert a.ingestion().build().ok
    assert len(client.calls) == 2  # the build classifies and extracts live


# --------------------------------------------------------------------------- #
# LLM router                                                                   #
# --------------------------------------------------------------------------- #


def _router_setup(
    tmp_path: Path, respond: Any, params: str = "{level: document}"
) -> tuple[Any, FakeClient]:
    enrichers = (
        "      - impl: regex_fields\n"
        "        params:\n"
        "          locale: it\n"
        "          fields:\n"
        "            importo_totale: {pattern: 'Totale documento: € ([0-9.,]+)', type: float}\n"
        "            fornitore: {pattern: 'Fornitore ([A-Z][a-z]+)', type: str}"
    )
    a, client = _setup(
        tmp_path,
        respond,
        enrichers=enrichers,
        router="llm",
        router_params=params,
    )
    assert a.ingestion().build().ok
    client.calls.clear()
    return a, client


class TestLLMRouter:
    def test_the_rules_answer_what_they_can_without_a_call(self, tmp_path: Path) -> None:
        a, client = _router_setup(tmp_path, lambda p: AssertionError("no call expected"))
        resp = a.query_engine().query("fatture con importo_totale superiore a 1000")
        assert str(resp.decision.path) == RoutePath.STRUCTURED
        assert resp.decision.router == "llm:rules" and not client.calls

    def test_a_paraphrase_becomes_a_structured_query(self, tmp_path: Path) -> None:
        def respond(params: dict[str, Any]) -> Any:
            schema = schema_of(params)
            assert schema["properties"]["filters"]["items"]["properties"]["field"]["enum"] == [
                "fornitore",
                "importo_totale",
            ]
            assert "importo_totale (float)" in prompt_of(params)
            return message(
                data={
                    "path": "structured",
                    "query_type": "numeric",
                    "standalone_question": "Quanto abbiamo speso con Rossi?",
                    "filters": [{"field": "fornitore", "op": "eq", "value": "Rossi"}],
                    "aggregation": {"op": "sum", "field": "importo_totale", "distinct": False},
                    "group_by": [],
                    "sub_queries": [],
                    "reason": "a total over one supplier",
                },
                model="claude-opus-5",
            )

        a, client = _router_setup(tmp_path, respond)
        resp = a.query_engine().query("Quanto abbiamo speso con Rossi?")
        assert len(client.calls) == 1
        assert str(resp.decision.path) == RoutePath.STRUCTURED
        assert resp.decision.router == "llm"
        assert (
            resp.records is not None and resp.records.as_dicts()[0]["sum_importo_totale"] == 1250.0
        )
        assert resp.cost_usd > 0

    def test_a_route_that_fails_its_checks_falls_back_to_the_rules(self, tmp_path: Path) -> None:
        answer = {
            "path": "structured",
            "query_type": "numeric",
            "standalone_question": "x",
            "filters": [{"field": "importo_totale", "op": "gt", "value": "molti"}],
            "aggregation": None,
            "group_by": [],
            "sub_queries": [],
            "reason": "",
        }
        a, _ = _router_setup(tmp_path, lambda p: message(data=answer))
        resp = a.query_engine().query("le fatture grosse di Rossi")
        assert resp.decision.router == "llm:rules"
        assert "llm route rejected" in resp.decision.reason
        assert "'molti' is not a float" in resp.decision.reason

    def test_a_follow_up_is_made_standalone_for_retrieval(self, tmp_path: Path) -> None:
        def respond(params: dict[str, Any]) -> Any:
            assert "<conversation>\nChi è il fornitore della fattura 42?" in prompt_of(params)
            return message(
                data={
                    "path": "lookup",
                    "query_type": "factual",
                    "standalone_question": "Quali sono i termini di pagamento della fattura 42?",
                    "filters": [],
                    "aggregation": None,
                    "group_by": [],
                    "sub_queries": [],
                    "reason": "a follow-up about the same invoice",
                }
            )

        a, _ = _router_setup(tmp_path, respond)
        resp = a.query_engine().query(
            "e i termini di pagamento?", context=("Chi è il fornitore della fattura 42?",)
        )
        assert (
            resp.decision.rewritten_query == "Quali sono i termini di pagamento della fattura 42?"
        )
        assert resp.hits and "Pagamento a 30 giorni" in resp.hits[0].unit.unit.text
        log = (Path(a.paths.store) / "route-decisions.jsonl").read_text().splitlines()
        assert json.loads(log[-1])["rewritten_query"].startswith("Quali sono")

    def test_an_unreachable_model_falls_back_to_the_rules(self, tmp_path: Path) -> None:
        a, _ = _router_setup(tmp_path, lambda p: ConnectionError("no route to host"))
        resp = a.query_engine().query("chi è il fornitore della fattura 42?")
        assert str(resp.decision.path) == RoutePath.LOOKUP
        assert "llm router failed" in resp.decision.reason

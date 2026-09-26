"""Documents read together: an attachment beside the message it came in.

A test report, an offer's revision, a signed annex often say what they are
about only in the email they were sent with. An enricher scoped ``RELATED``
reads those documents too, which makes them inputs like any other:

* part of its cache key;
* recorded in the ledger, so a changed message restages its attachments though
  their bytes did not change;
* never read beside a document with other readers, so nothing written about one
  carries the other's text to someone who may not read it;
* and, at query time, a reading that draws on a document the caller may not see
  is not shown.
"""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path
from typing import Any

from indexer.agent import AgentTools
from indexer.core.document import PARENT_KEY, ParsedDocument, SourceDocument
from indexer.core.ids import ContentHash, DocumentId, UnitId
from indexer.core.ledger import ChangeKind
from indexer.core.provenance import Provenance, Span
from indexer.core.stages import RelatedDocument, enrich_input_hash
from indexer.core.unit import ContextScope, Unit
from indexer.impls.enrich_resolve import LLMResolver
from indexer.pipeline import assemble
from indexer.pipeline.ingest import _Family
from llm_fakes import FakeClient
from test_resolver import ATTACHED, _document_of, _reader

VERBALE = """\
VERBALE DI PROVA

La prova di tenuta è stata eseguita alla pressione di 6 bar per 24 ore con esito positivo.
"""


def _message(site: str) -> str:
    msg = EmailMessage()
    msg["Subject"] = "Collaudo"
    msg["From"] = "Luca Verdi <l.verdi@rossiimpianti.it>"
    msg["To"] = "archivio@bianchilogistica.it"
    msg["Date"] = "Tue, 04 Mar 2025 10:00:00 +0100"
    msg.set_content(
        "Buongiorno,\n\nin allegato il verbale di collaudo dell'impianto idrico-sanitario "
        f"del cantiere di {site}.\n"
    )
    msg.add_attachment(VERBALE.encode(), maintype="text", subtype="plain", filename="verbale.txt")
    return msg.as_string()


CONFIG = """
schema_version: 1
project: {{name: archivio}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params: {{root: {data}, include: ["**/*.eml"], expand: [eml], default_acl: ["group:a"]}}
ingestion:
  parse:
    enabled: true
    routes:
      - {{when: {{media_type: message/rfc822}}, impl: email, params: {{labels: it}}}}
    default: {{impl: text}}
  segment: {{impl: structural, max_tokens: 400}}
  enrich:
    enabled: true
    enrichers:
      - {{impl: llm_resolver, params: {{relations: [container]}}}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25, params: {{fallback_language: it}}}}
query:
  route:
    enabled: false
    paths:
      structured: {{targets: []}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
  access: {{enabled: true}}
"""


def _setup(tmp_path: Path, site: str = "Via Roma 10") -> tuple[Path, Any, FakeClient]:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "collaudo.eml").write_text(_message(site), encoding="utf-8", newline="\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG.format(root=tmp_path.as_posix(), data=data.as_posix()), encoding="utf-8")
    client = FakeClient(_reader(ATTACHED))
    a = assemble(cfg, llm_client=client)
    assert a.ingestion().build().ok
    return cfg, a, client


def _attachment_calls(client: FakeClient) -> list[dict[str, Any]]:
    return [c for c in client.calls if "La prova di tenuta" in _document_of(c)]


def test_a_changed_message_restages_the_attachment_it_carries(tmp_path: Path) -> None:
    cfg, _, client = _setup(tmp_path)
    (first,) = _attachment_calls(client)
    assert "Via Roma 10" in first["messages"][0]["content"][0]["text"]

    # Unchanged: nothing to do, nothing asked.
    again = FakeClient(_reader(ATTACHED))
    pipeline = assemble(cfg, llm_client=again).ingestion()
    assert {c.kind for c in pipeline.plan()} == {ChangeKind.UNCHANGED}

    # The message changes; the attachment's bytes do not.
    (tmp_path / "data" / "collaudo.eml").write_text(
        _message("Via Roma 12"), encoding="utf-8", newline="\n"
    )
    plan = {c.document_id: c for c in pipeline.plan()}
    restaged = [c for c in plan.values() if c.kind is ChangeKind.RESTAGED]
    (attachment,) = restaged
    assert attachment.reason == "a document it is read with changed"
    assert attachment.stages == ("enrich:llm_resolver",)
    assert pipeline.build().ok
    (second,) = _attachment_calls(again)
    assert "Via Roma 12" in second["messages"][0]["content"][0]["text"]


def test_a_reading_that_draws_on_a_document_the_caller_may_not_see_is_not_shown(
    tmp_path: Path,
) -> None:
    _, a, _ = _setup(tmp_path)
    query = "prova di tenuta impianto idrico-sanitario Via Roma 10"
    tools = AgentTools(a, principals=["group:a"])
    top = tools.search(query)["results"][0]
    assert top["resolved"][0]["refers_to"][0]["relation"] == "container"

    container = top["resolved"][0]["refers_to"][0]["document_id"]
    visible = tools._visible
    tools._visible = lambda eu: str(eu.document_id) != container and visible(eu)  # type: ignore[method-assign]
    hidden = tools.search(query)["results"]
    shown = next(r for r in hidden if r["unit_id"] == top["unit_id"])
    assert "resolved" not in shown


# --------------------------------------------------------------------------- #
# which documents, and what the key holds                                      #
# --------------------------------------------------------------------------- #


class _Src(SourceDocument):
    __slots__ = ()


def _src(doc_id: str, parent: str | None = None, acl: str = "group:a") -> SourceDocument:
    meta: dict[str, Any] = {"acl": [acl]}
    if parent is not None:
        meta[PARENT_KEY] = parent
    return _Src(
        document_id=DocumentId(doc_id),
        source_uri=f"mem://{doc_id}",
        content_hash=ContentHash(f"h-{doc_id}"),
        media_type="text/plain",
        size_bytes=1,
        metadata=meta,
    )


def test_relations_come_from_the_scanner_s_parent_links() -> None:
    mail, a1, a2 = _src("mail"), _src("a1", "mail"), _src("a2", "mail")
    family = _Family({d.document_id: d for d in (mail, a1, a2)})
    kinds = ("container", "sibling", "attachment")
    assert [(r, d.document_id) for r, d in family.related(a1, kinds)] == [
        ("container", "mail"),
        ("sibling", "a2"),
    ]
    assert [(r, d.document_id) for r, d in family.related(mail, kinds)] == [
        ("attachment", "a1"),
        ("attachment", "a2"),
    ]
    assert family.related(a1, ("attachment",)) == []


def test_a_document_is_never_read_beside_one_with_other_readers(tmp_path: Path) -> None:
    _, a, _ = _setup(tmp_path)
    pipeline = a.ingestion()
    mail, same, other = _src("mail"), _src("a1", "mail"), _src("a2", "mail", acl="group:hr")
    family = _Family({d.document_id: d for d in (mail, same, other)})
    assert pipeline.access_field == "acl"
    assert [d.document_id for _, d in pipeline._linked(same, family, ("container",))] == ["mail"]
    assert pipeline._linked(other, family, ("container",)) == []
    pipeline.access_field = None
    assert len(pipeline._linked(other, family, ("container",))) == 1


def _parsed(doc_id: str, text: str) -> ParsedDocument:
    from indexer.core.ids import hash_text

    return ParsedDocument(
        document_id=DocumentId(doc_id),
        source_uri=f"mem://{doc_id}",
        text=text,
        blocks=(),
        source_hash=hash_text(text),
    )


def test_what_it_read_of_other_documents_is_in_its_key() -> None:
    doc = _parsed("a1", "La prova di tenuta.")
    unit = Unit(
        unit_id=UnitId("u"),
        document_id=doc.document_id,
        text=doc.text,
        provenance=Provenance(document_id=doc.document_id, span=Span(0, len(doc.text))),
    )
    before = [RelatedDocument("container", _parsed("mail", "Cantiere di Via Roma 10."))]
    after = [RelatedDocument("container", _parsed("mail", "Cantiere di Via Roma 12."))]
    reads = LLMResolver({"model": "m", "relations": ["container"]})
    assert reads.scope == ContextScope.RELATED
    assert enrich_input_hash(reads, unit, doc, {}, before) != enrich_input_hash(
        reads, unit, doc, {}, after
    )
    alone = LLMResolver({"model": "m"})
    assert enrich_input_hash(alone, unit, doc, {}, before) == enrich_input_hash(
        alone, unit, doc, {}, after
    )

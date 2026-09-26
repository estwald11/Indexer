"""Passages whose meaning is written somewhere else.

Each of these means more than its words say, so no index finds it by what it
means:

* a specification item that says "Idem c.s., ma per vuotatoi";
* a reply that says "va bene, procediamo con la seconda soluzione";
* a clause that says "L'Appaltatore risponde dei ritardi nei termini
  dell'art. 12";
* a maintenance paragraph that opens with "Esso";
* a test report whose site and system are named only in the email it came
  with.

What is tested, across those kinds of document:

* a model's reading of such a statement, checked against the texts it names
  before it is indexed;
* what the call sends, and what it costs;
* what an agent is handed;
* what shaping may merge.

A fake client answers.
"""

from __future__ import annotations

import re
from dataclasses import replace
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pytest

from indexer.agent import DERIVED, AgentTools
from indexer.cli import review_items
from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.document import ParsedDocument, SourceDocument
from indexer.core.errors import ConfigError
from indexer.core.ids import DocumentId, UnitId, hash_text
from indexer.core.provenance import Provenance, Span
from indexer.core.stages import EnrichContext, StageContext
from indexer.core.unit import ContextScope, Unit
from indexer.impls.enrich_resolve import LLMResolver, ungrounded
from indexer.impls.parse import TextParser
from indexer.impls.segment import ItemSegmenter
from indexer.pipeline import assemble
from llm_fakes import FakeClient, message, schema_of

CTX = StageContext(cache=NullCache(), accountant=InMemoryAccountant())

# --------------------------------------------------------------------------- #
# an archive of five kinds of document                                         #
# --------------------------------------------------------------------------- #

CAPITOLATO = """\
Capitolato speciale d'appalto. Impianti meccanici, specifiche tecniche.

03 IMPIANTO IDRICO-SANITARIO

03.01.003 Vuotatoio. Fornitura e posa in opera di vuotatoio in vitreous china bianco con
griglia ribaltabile in acciaio inox, scarico a parete DN 110, cassetta di risciacquo esterna.

03.02 TELAI DI SOSTEGNO

03.02.001 Telaio per lavabo. Fornitura e posa in opera di telaio di sostegno autoportante
per lavabo sospeso, tipo Geberit Duofix o equivalente, in acciaio zincato, piedini
regolabili, barre filettate M10, curva di scarico in PE DN 40/50, isolamento acustico.
03.02.002 Idem c.s., ma per vuotatoi.
03.02.003 Telaio per vaso sospeso tipo Geberit Duofix con cassetta di risciacquo da
incasso, curva di scarico DN 90, barre filettate M12.

03.03 TUBAZIONI

03.03.001 Tubazioni multistrato. Fornitura e posa in opera di tubazioni in multistrato
PEX-AL-PEX per acqua fredda e calda sanitaria, pre-isolate, con raccordi a pressare.
Diametro 16x2 mm.
03.03.002 Idem, diametro 26x3 mm.
"""

CONTRATTO = """\
# Contratto di manutenzione

## Art. 1 - Definizioni

Nel presente contratto si intende per «Appaltatore» la società Rossi Impianti S.r.l.,
con sede in Brescia, e per «Committente» la società Bianchi Logistica S.p.A.,
proprietaria degli immobili di Bergamo e di Lodi.

## Art. 12 - Penali

Per ogni giorno di ritardo nell'intervento su guasto rispetto ai tempi dell'allegato B
è applicata una penale di 150 euro, fino a un massimo del 10% del canone annuo.

## Art. 15 - Responsabilità

L'Appaltatore risponde dei ritardi nei termini dell'art. 12 e tiene indenne il
Committente da ogni pretesa di terzi derivante dagli interventi eseguiti.
"""

RELAZIONE = """\
# Relazione tecnica, riqualificazione della sede di Lodi

## Centrale frigorifera

Il nuovo gruppo frigorifero condensato ad aria da 250 kW sostituisce le due macchine
esistenti. È installato sulla copertura del corpo B, su basamento antivibrante, e
alimenta il circuito dei ventilconvettori.

## Manutenzione

Esso dovrà essere sottoposto a manutenzione semestrale da parte di un tecnico
abilitato, con verifica delle fughe di refrigerante e registrazione degli interventi
nel libretto di impianto.
"""

RISPOSTA = """\
Buongiorno Anna,

va bene, procediamo con la seconda soluzione. Attendiamo la conferma d'ordine entro venerdì.

Mario

Il giorno lun 3 mar 2025 alle 09:15 Anna Bianchi ha scritto:
> Buongiorno, vi proponiamo due soluzioni per la climatizzazione degli uffici di Bergamo:
> 1) sistema VRF con dodici unità interne, 48.500 euro;
> 2) pompa di calore aria-acqua Mitsubishi da 14 kW con ventilconvettori, 36.900 euro.
"""

ACCOMPAGNATORIA = """\
Buongiorno,

in allegato il verbale di collaudo dell'impianto idrico-sanitario del cantiere di
Via Roma 10, piano interrato.

Cordiali saluti,
Luca Verdi
"""

VERBALE = """\
VERBALE DI PROVA

La prova di tenuta è stata eseguita alla pressione di 6 bar per 24 ore con esito positivo.
Non sono state rilevate perdite né cali di pressione.
"""


def _email(subject: str, sender: str, body: str, attachment: tuple[str, str] | None = None) -> str:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = "archivio@bianchilogistica.it"
    msg["Date"] = "Tue, 04 Mar 2025 10:00:00 +0100"
    msg.set_content(body)
    if attachment is not None:
        name, text = attachment
        msg.add_attachment(text.encode("utf-8"), maintype="text", subtype="plain", filename=name)
    return msg.as_string()


ARCHIVE = {
    "capitolato.txt": CAPITOLATO,
    "contratto.md": CONTRATTO,
    "relazione.md": RELAZIONE,
    "risposta.eml": _email(
        "Re: Climatizzazione uffici di Bergamo", "Mario Rossi <m.rossi@bl.it>", RISPOSTA
    ),
    "collaudo.eml": _email(
        "Cantiere Via Roma 10",
        "Luca Verdi <l.verdi@rossiimpianti.it>",
        ACCOMPAGNATORIA,
        ("verbale.txt", VERBALE),
    ),
}

# What a model reads each of them as.
FRAME = {
    "quote": "Idem c.s., ma per vuotatoi",
    "refers_to": ["03.02.001 Telaio per lavabo"],
    "standalone": (
        "Telaio di sostegno autoportante per vuotatoi, tipo Geberit Duofix o equivalente, "
        "in acciaio zincato, piedini regolabili, barre filettate M10, curva di scarico in "
        "PE DN 40/50, isolamento acustico."
    ),
}
PIPE = {
    "quote": "Idem, diametro 26x3 mm",
    "refers_to": ["03.03.001 Tubazioni multistrato"],
    "standalone": (
        "Tubazioni in multistrato PEX-AL-PEX per acqua fredda e calda sanitaria, "
        "pre-isolate, con raccordi a pressare. Diametro 26x3 mm."
    ),
}
CLAUSE = {
    "quote": "L'Appaltatore risponde dei ritardi nei termini dell'art. 12",
    "refers_to": [
        "Nel presente contratto si intende per «Appaltatore»",
        "Per ogni giorno di ritardo nell'intervento su guasto",
    ],
    "standalone": (
        "Rossi Impianti S.r.l. risponde dei ritardi nell'intervento su guasto con una penale "
        "di 150 euro per ogni giorno di ritardo, fino a un massimo del 10% del canone annuo."
    ),
}
PRONOUN = {
    "quote": "Esso dovrà essere sottoposto a manutenzione semestrale",
    "refers_to": ["Il nuovo gruppo frigorifero condensato ad aria"],
    "standalone": (
        "Il gruppo frigorifero condensato ad aria da 250 kW dovrà essere sottoposto a "
        "manutenzione semestrale da parte di un tecnico abilitato."
    ),
}
REPLY = {
    "quote": "va bene, procediamo con la seconda soluzione",
    "refers_to": ["vi proponiamo due soluzioni per la climatizzazione"],
    "standalone": (
        "Va bene, procediamo con la seconda soluzione: pompa di calore aria-acqua "
        "Mitsubishi da 14 kW con ventilconvettori, 36.900 euro."
    ),
}
ATTACHED = {
    "quote": "La prova di tenuta è stata eseguita alla pressione di 6 bar per 24 ore",
    "refers_to": ["in allegato il verbale di collaudo dell'impianto idrico-sanitario"],
    "standalone": (
        "La prova di tenuta dell'impianto idrico-sanitario del cantiere di Via Roma 10 è "
        "stata eseguita alla pressione di 6 bar per 24 ore con esito positivo."
    ),
}
READINGS = (FRAME, PIPE, CLAUSE, PRONOUN, REPLY, ATTACHED)

CONFIG = """
schema_version: 1
project: {{name: archivio}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params:
        root: {data}
        include: ["**/*.txt", "**/*.md", "**/*.eml"]
        expand: [eml]
ingestion:
  parse:
    enabled: true
    routes:
      - {{when: {{media_type: message/rfc822}}, impl: email, params: {{labels: it}}}}
      - {{when: {{media_type: text/markdown}}, impl: markdown}}
    default: {{impl: text}}
  segment: {{impl: items, max_tokens: 400}}
  enrich:
    enabled: true
    batch_size: {batch_size}
    enrichers:
      - {{impl: llm_resolver, enabled: {resolver}, params: {params}}}
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
  shape: {{enabled: true, distinguish_by: {distinguish}, expand_neighbors: {neighbors}}}
"""

_PASSAGE = re.compile(r'<passage id="(p\d+)">\n(.*?)\n</passage>', re.S)


def _blocks(params: dict[str, Any]) -> list[str]:
    return [b["text"] for b in params["messages"][0]["content"]]


def _document_of(params: dict[str, Any]) -> str:
    return next(b for b in _blocks(params) if b.startswith("<document"))


def _reader(*readings: dict[str, Any], by_document: dict[str, dict[str, Any]] | None = None) -> Any:
    """A resolver's answers. Each passage the call asks about, whether marked in
    the document or listed after it, gets the readings whose quote it holds --
    or whose ``_at`` it holds, for a reading that misquotes. ``by_document``
    adds readings for the documents containing its keys."""

    def respond(params: dict[str, Any]) -> Any:
        passages = dict(_PASSAGE.findall("\n".join(_blocks(params))))
        document = _document_of(params)
        pool = list(readings)
        for marker, reading in (by_document or {}).items():
            if marker in document:
                pool.append(reading)
        answer = {
            pid: [
                {k: v for k, v in r.items() if k != "_at"}
                for r in pool
                if r.get("_at", r["quote"]) in passages.get(pid, "")
            ]
            for pid in schema_of(params)["properties"]
        }
        return message(data=answer, model=params["model"])

    return respond


def _setup(
    tmp_path: Path,
    respond: Any,
    *,
    files: dict[str, str] | None = None,
    resolver: bool = True,
    params: str = "{relations: [container]}",
    batch_size: int = 16,
    distinguish: str = "[llm_resolver]",
    neighbors: int = 0,
) -> tuple[Any, FakeClient]:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, body in (files or {"capitolato.txt": CAPITOLATO}).items():
        (data / name).write_text(body, encoding="utf-8", newline="\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        CONFIG.format(
            root=tmp_path.as_posix(),
            data=data.as_posix(),
            resolver=str(resolver).lower(),
            params=params,
            batch_size=batch_size,
            distinguish=distinguish,
            neighbors=neighbors,
        ),
        encoding="utf-8",
    )
    client = FakeClient(respond)
    a = assemble(cfg, llm_client=client)
    assert a.ingestion().build().ok
    return a, client


def _units(a: Any) -> list[Any]:
    store = a.unit_store
    return sorted(
        (store.get(uid) for uid in store.all_ids()),
        key=lambda eu: (str(eu.unit.metadata.get("relpath")), eu.unit.ordinal),
    )


def _unit(a: Any, text: str) -> Any:
    return next(eu for eu in _units(a) if text in eu.unit.text)


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The archive indexed twice: as written, and with the resolver."""
    root = tmp_path_factory.mktemp("archive")
    before, _ = _setup(root / "before", _reader(*READINGS), files=ARCHIVE, resolver=False)
    after, client = _setup(root / "after", _reader(*READINGS), files=ARCHIVE)
    return {"before": before, "after": after, "client": client}


# --------------------------------------------------------------------------- #
# found by what it means, in every kind of document                            #
# --------------------------------------------------------------------------- #

CASES = [
    # (query, what the passage says, its reading)
    ("telaio autoportante per vuotatoio", "03.02.002 Idem c.s.", FRAME),
    ("penale per i ritardi di Rossi Impianti", "L'Appaltatore risponde", CLAUSE),
    ("manutenzione del gruppo frigorifero da 250 kW", "Esso dovrà", PRONOUN),
    ("pompa di calore Mitsubishi", "procediamo con la seconda", REPLY),
    ("prova di tenuta impianto idrico-sanitario Via Roma 10", "La prova di tenuta", ATTACHED),
]


@pytest.mark.parametrize(
    ("query", "says", "reading"), CASES, ids=["item", "clause", "pronoun", "reply", "attachment"]
)
def test_a_passage_is_found_by_what_it_means(
    archive: dict[str, Any], query: str, says: str, reading: dict[str, Any]
) -> None:
    before = AgentTools(archive["before"]).search(query)["results"]
    assert not before or says not in before[0]["text"]

    found = AgentTools(archive["after"]).search(query)
    top = found["results"][0]
    assert says in top["text"]
    (resolved,) = top["resolved"]
    assert resolved["reads_as"] == reading["standalone"]
    assert resolved["quote"] == reading["quote"]
    assert found["derived_text"] == DERIVED
    # The passage's own text is never rewritten.
    eu = archive["after"].unit_store.get(UnitId(top["unit_id"]))
    assert reading["standalone"] not in eu.unit.text
    assert eu.indexing_text().startswith(reading["standalone"])


def test_a_reply_is_not_found_by_the_message_it_quotes_until_it_is_read(
    archive: dict[str, Any],
) -> None:
    # The history is in the document, for the resolver to read -- not in any
    # unit, or the reply would match everything the quoted message says.
    before = AgentTools(archive["before"]).search("sistema VRF con dodici unità interne")
    assert all("Mario" not in r["text"] for r in before["results"])
    top = AgentTools(archive["after"]).search("pompa di calore Mitsubishi")["results"][0]
    (ref,) = top["resolved"][0]["refers_to"]
    assert ref["relation"] == "quoted" and ref["unit_id"] is None
    assert ref["quote"] == "vi proponiamo due soluzioni per la climatizzazione"
    assert "pompa di calore aria-acqua Mitsubishi" in ref["text"]


def test_an_attachment_is_read_beside_the_message_it_came_in(archive: dict[str, Any]) -> None:
    a = archive["after"]
    top = AgentTools(a).search(CASES[4][0])["results"][0]
    (ref,) = top["resolved"][0]["refers_to"]
    email = _unit(a, "in allegato il verbale di collaudo")
    assert ref["relation"] == "container"
    assert ref["document_id"] == str(email.document_id) != top["document_id"]
    assert ref["unit_id"] == str(email.unit_id)


def test_each_text_a_reading_draws_on_is_cited(archive: dict[str, Any]) -> None:
    a = archive["after"]
    top = AgentTools(a).search(CASES[1][0])["results"][0]
    refs = top["resolved"][0]["refers_to"]
    assert [r["quote"] for r in refs] == CLAUSE["refers_to"]
    assert [r["unit_id"] for r in refs] == [
        str(_unit(a, "si intende per «Appaltatore»").unit_id),
        str(_unit(a, "Per ogni giorno di ritardo").unit_id),
    ]
    idem = AgentTools(a).search(CASES[0][0])["results"][0]
    (ref,) = idem["resolved"][0]["refers_to"]
    assert ref["unit_id"] == str(_unit(a, "03.02.001 Telaio per lavabo").unit_id)


# --------------------------------------------------------------------------- #
# what the call sends                                                          #
# --------------------------------------------------------------------------- #


def _call_for(client: FakeClient, text: str) -> dict[str, Any]:
    return next(c for c in client.calls if text in _document_of(c))


class TestTheCall:
    def test_the_instructions_are_the_same_for_every_call_and_cached(
        self, archive: dict[str, Any]
    ) -> None:
        systems = [c["system"] for c in archive["client"].calls]
        assert len(systems) == 6  # one per document: every one fits a batch
        assert all(s == systems[0] for s in systems)
        (block,) = systems[0]
        assert block["cache_control"] == {"type": "ephemeral"}
        assert "from text outside their passage" in block["text"]

    def test_a_document_is_sent_once_with_its_passages_marked_in_place(
        self, archive: dict[str, Any]
    ) -> None:
        call = _call_for(archive["client"], "03.02.001 Telaio per lavabo")
        document, question = _blocks(call)
        ids = list(schema_of(call)["properties"])
        assert ids == [f"p{i}" for i in range(1, len(ids) + 1)]
        assert schema_of(call)["required"] == ids
        assert dict(_PASSAGE.findall(document))["p4"] == "03.02.002 Idem c.s., ma per vuotatoi."
        # Named, not repeated.
        assert "Idem c.s." not in question and ", ".join(ids) in question
        # One batch is the whole document: nothing would read it from the cache.
        assert "cache_control" not in call["messages"][0]["content"][0]

    def test_what_a_reply_quotes_is_read_as_quoted(self, archive: dict[str, Any]) -> None:
        document = _document_of(_call_for(archive["client"], "procediamo con la seconda"))
        history = document.split("<quoted>\n", 1)[1]
        assert history.startswith("Il giorno lun 3 mar 2025 alle 09:15 Anna Bianchi ha scritto:")
        assert "pompa di calore aria-acqua Mitsubishi" in history
        assert "procediamo" not in history

    def test_an_attachment_is_sent_after_the_message_it_came_in(
        self, archive: dict[str, Any]
    ) -> None:
        call = _call_for(archive["client"], "La prova di tenuta")
        related, document, _ = _blocks(call)
        assert related.startswith('<related_document id="r1" relation="container"')
        assert "in allegato il verbale di collaudo" in related
        assert "La prova di tenuta" not in related and "in allegato" not in document
        # The message itself has no container to read.
        email = _call_for(archive["client"], "in allegato il verbale di collaudo")
        assert len(_blocks(email)) == 2

    def test_a_long_document_is_cached_for_its_later_batches(self, tmp_path: Path) -> None:
        _, client = _setup(tmp_path, _reader(FRAME, PIPE), batch_size=3)
        assert len(client.calls) == 3
        documents = [c["messages"][0]["content"][0] for c in client.calls]
        assert all(d["cache_control"] == {"type": "ephemeral"} for d in documents)
        assert len({d["text"] for d in documents}) == 1
        asked = [list(schema_of(c)["properties"]) for c in client.calls]
        assert asked == [["p1", "p2", "p3"], ["p4", "p5", "p6"], ["p7"]]

    def test_a_long_document_is_read_up_to_the_batch(self, tmp_path: Path) -> None:
        longer = CAPITOLATO + (
            "\n03.04 VALVOLE\n\n03.04.001 Valvola di bilanciamento statica con attacchi "
            "filettati e prese di pressione.\n"
        )
        a, client = _setup(
            tmp_path,
            _reader(FRAME),
            files={"capitolato.txt": longer},
            params="{max_document_chars: 1000}",
            batch_size=3,
        )
        assert len(longer) > 1000 and len(client.calls) == 3
        for call in client.calls:
            document = _document_of(call)
            window = re.search(r"characters (\d+) to (\d+) of", document)
            assert window is not None
            lo, hi = map(int, window.groups())
            assert hi - lo <= 1000
            assert all(f'<passage id="{pid}">' in document for pid in schema_of(call)["properties"])
            # A window is not the same document twice: not worth caching.
            assert "cache_control" not in call["messages"][0]["content"][0]
        # Read against the whole document all the same.
        assert _unit(a, "03.02.002").indexing_text().startswith("Telaio di sostegno")

    def test_overlapping_passages_are_listed_instead(self) -> None:
        text = "Tabella prezzi\n\n| voce | prezzo |\n| telaio | 120 |\n| idem per vuotatoi | 130 |"
        doc = _parsed(text)
        span = Span(16, len(text))
        pieces = [_unit_at(doc, i, span, text[16:]) for i in range(2)]  # a split table
        ctx = EnrichContext(document=doc, units=pieces, stage=CTX)
        ((_, params),) = LLMResolver({"model": "claude-opus-5"}).requests_for(pieces, ctx)
        document, question = _blocks(params)
        assert "<passage" not in document
        assert question.count("<passage id=") == 2

    def test_a_document_configured_out_is_not_sent(self, tmp_path: Path) -> None:
        files = {"capitolato.txt": CAPITOLATO, "contratto.md": CONTRATTO}
        a, client = _setup(
            tmp_path,
            _reader(FRAME, CLAUSE),
            files=files,
            params="{skip_when: {name: contratto.md}}",
        )
        assert len(client.calls) == 1 and "Telaio per lavabo" in _document_of(client.calls[0])
        clause = _unit(a, "L'Appaltatore risponde")
        assert clause.indexing_text() == clause.unit.text
        assert "llm_resolver" in clause.enrichments

    def test_the_call_is_accounted_once(self, tmp_path: Path) -> None:
        a, client = _setup(tmp_path, _reader(FRAME, PIPE))
        costs = [eu.enrichments["llm_resolver"].cost_usd for eu in _units(a)]
        assert len(client.calls) == 1 and costs[0] > 0 and not any(costs[1:])

    def test_relations_it_does_not_know_are_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="relations"):
            _setup(tmp_path, _reader(), params="{relations: [thread]}")

    def test_reading_related_documents_widens_its_scope(self) -> None:
        assert LLMResolver({"model": "m"}).scope == ContextScope.DOCUMENT
        related = LLMResolver({"model": "m", "relations": ["container"]})
        assert related.scope == ContextScope.RELATED and related.relations == ("container",)


def _parsed(text: str) -> ParsedDocument:
    doc = _Doc(
        document_id=DocumentId("d"),
        source_uri="mem://d",
        content_hash=hash_text(text),
        media_type="text/plain",
        size_bytes=len(text),
        metadata={"body": text},
    )
    return TextParser({}).parse(doc, CTX)


def _unit_at(doc: ParsedDocument, i: int, span: Span, text: str) -> Unit:
    return Unit(
        unit_id=UnitId(f"u{i}"),
        document_id=doc.document_id,
        text=text,
        provenance=Provenance(document_id=doc.document_id, span=span, source_uri="mem://d"),
        ordinal=i,
        verbatim=False,
    )


class _Doc(SourceDocument):
    __slots__ = ()

    def load(self) -> bytes:
        return self.metadata["body"].encode()  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# a reading is checked before it is indexed                                    #
# --------------------------------------------------------------------------- #


class TestChecks:
    def test_a_reading_that_adds_a_figure_is_held_for_review(self, tmp_path: Path) -> None:
        # DN 110 is what a slop sink's outlet is, three items up. It is not
        # what "idem" takes from the frame for washbasins.
        wrong = {**FRAME, "standalone": "Telaio di sostegno per vuotatoi, curva di scarico DN 110."}
        a, _ = _setup(tmp_path, _reader(wrong, PIPE))
        idem = _unit(a, "03.02.002")
        assert idem.indexing_text() == idem.unit.text
        (rejected,) = idem.enrichments["llm_resolver"].extra["rejected"]
        assert "figures" in rejected["reason"] and "110" in rejected["reason"]
        items = [i for i in review_items(a) if i["enricher"] == "llm_resolver"]
        assert [i["field"] for i in items] == ["resolution"]
        # The other reading of the same call is kept.
        assert _unit(a, "03.03.002").indexing_text().startswith("Tubazioni in multistrato")

    def test_a_reading_that_names_what_its_sources_do_not_is_held(self, tmp_path: Path) -> None:
        wrong = {**REPLY, "standalone": "Va bene, procediamo con la pompa di calore Daikin."}
        a, _ = _setup(tmp_path, _reader(wrong), files={"r.eml": ARCHIVE["risposta.eml"]})
        (rejected,) = _unit(a, "procediamo").enrichments["llm_resolver"].extra["rejected"]
        assert "names" in rejected["reason"] and "Daikin" in rejected["reason"]

    @pytest.mark.parametrize(
        ("change", "reason"),
        [
            ({"quote": "Idem come sopra", "_at": FRAME["quote"]}, "not in the passage"),
            ({"refers_to": ["03.09.001 Telaio per bidet"]}, "not in the document"),
            ({"refers_to": []}, "names no text"),
            # A statement is not what it refers to.
            ({"refers_to": ["Idem c.s., ma per vuotatoi"]}, "not in the document"),
        ],
    )
    def test_a_reading_must_quote_what_it_reads(
        self, tmp_path: Path, change: dict[str, Any], reason: str
    ) -> None:
        a, _ = _setup(tmp_path, _reader({**FRAME, **change}))
        idem = _unit(a, "03.02.002")
        (rejected,) = idem.enrichments["llm_resolver"].extra["rejected"]
        assert reason in rejected["reason"]
        assert idem.indexing_text() == idem.unit.text

    def test_a_related_document_is_quoted_only_as_far_as_it_was_shown(self, tmp_path: Path) -> None:
        files = {"collaudo.eml": ARCHIVE["collaudo.eml"]}
        params = "{relations: [container], max_related_chars: 40}"
        a, client = _setup(tmp_path, _reader(ATTACHED), files=files, params=params)
        related = _blocks(_call_for(client, "La prova di tenuta"))[0]
        assert related.endswith("[...]\n</related_document>")
        (rejected,) = _unit(a, "La prova di tenuta").enrichments["llm_resolver"].extra["rejected"]
        assert "not in the document or a related one" in rejected["reason"]

    def test_the_same_line_in_two_places_is_read_in_each(self) -> None:
        parsed = _parsed(CAPITOLATO)
        unit = next(u for u in ItemSegmenter({}).segment(parsed, CTX) if "Idem c.s." in u.text)
        span = unit.provenance.span
        elsewhere = replace(
            unit, provenance=replace(unit.provenance, span=Span(span.start + 40, span.end + 40))
        )
        resolver = LLMResolver({"model": "claude-opus-5"})
        assert resolver.input_hash(unit, parsed, {}) == resolver.input_hash(unit, parsed, {})
        assert resolver.input_hash(unit, parsed, {}) != resolver.input_hash(elsewhere, parsed, {})


def test_a_reading_may_inflect_the_document_s_words_but_not_add_to_them() -> None:
    source = "03.02.001 Telaio per lavabo sospeso, tipo Geberit Duofix, barre filettate M10."
    item = "03.02.002 Idem c.s., ma per vuotatoi."
    reading = "Telaio per vuotatoio sospeso, tipo Geberit Duofix, barre M10."
    assert ungrounded(reading, f"{source}\n{item}") is None
    assert ungrounded("Telai per lavabi sospesi con barre filettate M10.", source) is None
    assert "figures" in (ungrounded("Telaio per lavabo, barre M12.", source) or "")
    assert "names" in (ungrounded("Telaio per lavabo tipo Grohe.", source) or "")
    words = ungrounded("Telaio per lavabo con cassetta, placca, rubinetto e miscelatore.", source)
    assert words is not None and "words" in words


# --------------------------------------------------------------------------- #
# what shaping may merge, and what the agent is handed                         #
# --------------------------------------------------------------------------- #

OTHER = """\
03.02.001 Staffa per lavabo. Fornitura e posa in opera di staffa di sostegno per lavabo
sospeso, in acciaio zincato, con tasselli.
03.02.002 Idem c.s., ma per vuotatoi.
03.02.003 Staffa per orinatoio in acciaio zincato.
"""
BRACKET = {
    "quote": "Idem c.s., ma per vuotatoi",
    "refers_to": ["03.02.001 Staffa per lavabo"],
    "standalone": "Staffa di sostegno per vuotatoi, in acciaio zincato, con tasselli.",
}


class TestShaping:
    @pytest.mark.parametrize(("distinguish", "shown"), [("[llm_resolver]", 2), ("[]", 1)])
    def test_the_same_words_under_different_antecedents_are_two_passages(
        self, tmp_path: Path, distinguish: str, shown: int
    ) -> None:
        respond = _reader(
            PIPE, by_document={"Staffa per lavabo": BRACKET, "Telaio per lavabo": FRAME}
        )
        files = {"capitolato.txt": CAPITOLATO, "altro.txt": OTHER}
        a, _ = _setup(tmp_path, respond, files=files, distinguish=distinguish)
        results = AgentTools(a).search("Idem c.s., ma per vuotatoi", top_k=10)["results"]
        idem = [r for r in results if r["text"].startswith("03.02.002")]
        assert len(idem) == shown
        if shown == 2:
            assert {r["resolved"][0]["reads_as"] for r in idem} == {
                FRAME["standalone"],
                BRACKET["standalone"],
            }

    def test_distinguish_by_names_a_configured_enricher(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="distinguish_by"):
            _setup(tmp_path, _reader(), distinguish="[llm_resolvr]")

    def test_the_agent_is_handed_the_neighbouring_text(self, tmp_path: Path) -> None:
        a, _ = _setup(tmp_path, _reader(FRAME, PIPE), neighbors=1)
        top = AgentTools(a).search("telaio autoportante per vuotatoio")["results"][0]
        assert top["neighbors"]["before"][0].startswith("03.02.001 Telaio per lavabo")
        assert top["neighbors"]["after"][0].startswith("03.02.003 Telaio per vaso")

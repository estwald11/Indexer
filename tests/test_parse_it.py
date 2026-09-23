"""Email, PEC and FatturaPA parsers -- and the archive they add up to.

The last test is the case this library is being adopted for: a PEC arrives
carrying a signed e-invoice; the archive is scanned, the envelope opened, the
invoice's exact figures land in the structured index, and an Italian question
about amounts is answered from them rather than from vector search.
"""

from __future__ import annotations

from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pytest

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.document import SourceDocument
from indexer.core.ids import DocumentId, hash_bytes
from indexer.core.query import RoutePath
from indexer.core.stages import StageContext
from indexer.eval.checks import check_parsed_document
from indexer.impls.parse_fatturapa import FatturaPAParser, read_fatturapa
from indexer.impls.parse_mail import EmailParser, strip_quoted
from indexer.pipeline import assemble

CTX = StageContext(cache=NullCache(), accountant=InMemoryAccountant())

FATTURA = """<?xml version="1.0" encoding="UTF-8"?>
<p:FatturaElettronica versione="FPR12"
  xmlns:p="http://ivaservizi.agenziaentrate.gov.it/docs/xsd/fatture/v1.2">
  <FatturaElettronicaHeader>
    <DatiTrasmissione>
      <IdTrasmittente><IdPaese>IT</IdPaese><IdCodice>01234567890</IdCodice></IdTrasmittente>
      <ProgressivoInvio>00001</ProgressivoInvio>
      <FormatoTrasmissione>FPR12</FormatoTrasmissione>
      <CodiceDestinatario>ABC1234</CodiceDestinatario>
    </DatiTrasmissione>
    <CedentePrestatore>
      <DatiAnagrafici>
        <IdFiscaleIVA><IdPaese>IT</IdPaese><IdCodice>01234567890</IdCodice></IdFiscaleIVA>
        <Anagrafica><Denominazione>Rossi Forniture Srl</Denominazione></Anagrafica>
        <RegimeFiscale>RF01</RegimeFiscale>
      </DatiAnagrafici>
      <Sede><Indirizzo>Via Roma</Indirizzo><NumeroCivico>1</NumeroCivico><CAP>20100</CAP>
        <Comune>Milano</Comune><Provincia>MI</Provincia><Nazione>IT</Nazione></Sede>
    </CedentePrestatore>
    <CessionarioCommittente>
      <DatiAnagrafici>
        <IdFiscaleIVA><IdPaese>IT</IdPaese><IdCodice>09876543210</IdCodice></IdFiscaleIVA>
        <Anagrafica><Denominazione>Bianchi Spa</Denominazione></Anagrafica>
      </DatiAnagrafici>
      <Sede><Indirizzo>Corso Italia 5</Indirizzo><CAP>10100</CAP><Comune>Torino</Comune>
        <Nazione>IT</Nazione></Sede>
    </CessionarioCommittente>
  </FatturaElettronicaHeader>
  <FatturaElettronicaBody>
    <DatiGenerali>
      <DatiGeneraliDocumento>
        <TipoDocumento>TD01</TipoDocumento><Divisa>EUR</Divisa><Data>2025-03-15</Data>
        <Numero>2025/123</Numero><ImportoTotaleDocumento>AMOUNT</ImportoTotaleDocumento>
        <Causale>Fornitura materiale d'ufficio</Causale>
      </DatiGeneraliDocumento>
      <DatiOrdineAcquisto><IdDocumento>ORD-77</IdDocumento></DatiOrdineAcquisto>
    </DatiGenerali>
    <DatiBeniServizi>
      <DettaglioLinee><NumeroLinea>1</NumeroLinea><Descrizione>Carta A4</Descrizione>
        <Quantita>100.00</Quantita><UnitaMisura>RISME</UnitaMisura>
        <PrezzoUnitario>5.00</PrezzoUnitario><PrezzoTotale>500.00</PrezzoTotale>
        <AliquotaIVA>22.00</AliquotaIVA></DettaglioLinee>
      <DettaglioLinee><NumeroLinea>2</NumeroLinea><Descrizione>Toner</Descrizione>
        <Quantita>10.00</Quantita><PrezzoUnitario>50.00</PrezzoUnitario>
        <PrezzoTotale>500.00</PrezzoTotale><AliquotaIVA>22.00</AliquotaIVA></DettaglioLinee>
      <DatiRiepilogo><AliquotaIVA>22.00</AliquotaIVA><ImponibileImporto>1000.00</ImponibileImporto>
        <Imposta>220.00</Imposta><EsigibilitaIVA>I</EsigibilitaIVA></DatiRiepilogo>
    </DatiBeniServizi>
    <DatiPagamento><CondizioniPagamento>TP02</CondizioniPagamento>
      <DettaglioPagamento><ModalitaPagamento>MP05</ModalitaPagamento>
        <DataScadenzaPagamento>2025-04-15</DataScadenzaPagamento>
        <ImportoPagamento>1220.00</ImportoPagamento><IBAN>IT60X0542811101000000123456</IBAN>
      </DettaglioPagamento>
    </DatiPagamento>
  </FatturaElettronicaBody>
</p:FatturaElettronica>
"""


class _Bytes(SourceDocument):
    __slots__ = ()

    def load(self) -> bytes:
        return self.metadata["_bytes"]  # type: ignore[no-any-return]


def _doc(name: str, data: bytes, media_type: str) -> SourceDocument:
    return _Bytes(
        document_id=DocumentId(name),
        source_uri=f"mem://{name}",
        content_hash=hash_bytes(data),
        media_type=media_type,
        size_bytes=len(data),
        metadata={"_bytes": data},
    )


def _invoice(amount: str = "1220.00") -> bytes:
    return FATTURA.replace("AMOUNT", amount).encode("utf-8")


class TestFatturaPA:
    def test_exact_facts_become_document_metadata(self) -> None:
        parsed = FatturaPAParser({}).parse(_doc("f.xml", _invoice(), "application/xml"), CTX)
        assert check_parsed_document(parsed) == []
        m = parsed.metadata
        assert m["tipo_documento"] == "fattura"
        assert m["numero_documento"] == "2025/123"
        assert m["data_documento"] == date(2025, 3, 15)
        assert m["importo_totale"] == 1220.0
        assert (m["imponibile"], m["imposta"]) == (1000.0, 220.0)
        assert m["cedente_piva"] == "IT01234567890"
        assert m["cessionario_denominazione"] == "Bianchi Spa"
        assert m["data_scadenza"] == date(2025, 4, 15)
        assert m["iban"] == ["IT60X0542811101000000123456"]
        assert m["modalita_pagamento"] == ["bonifico"]
        assert m["ordini_acquisto"] == ["ORD-77"]
        assert m["totali_coerenti"] is True

    def test_rendering_is_readable_italian_with_tables(self) -> None:
        parsed = FatturaPAParser({}).parse(_doc("f.xml", _invoice(), "application/xml"), CTX)
        assert parsed.blocks[0].text == "Fattura n. 2025/123 del 15/03/2025"
        assert "Importo totale: 1.220,00 EUR" in parsed.text
        tables = [b.table for b in parsed.blocks if b.table is not None]
        assert len(tables) == 3  # lines, VAT summary, payment
        assert tables[0].rows[1][1].text == "Carta A4"

    def test_an_inconsistent_total_is_flagged(self) -> None:
        parsed = FatturaPAParser({}).parse(
            _doc("f.xml", _invoice("1300.00"), "application/xml"), CTX
        )
        assert parsed.metadata["totali_coerenti"] is False

    def test_not_an_invoice_is_an_error_not_an_empty_document(self) -> None:
        with pytest.raises(ValueError, match="not a FatturaPA"):
            read_fatturapa(b"<ordine><numero>1</numero></ordine>")


class TestEmailParser:
    def _message(self, body: str) -> bytes:
        msg = EmailMessage()
        msg["Subject"] = "Re: Offerta 2025"
        msg["From"] = "Anna Bianchi <anna@azienda.it>"
        msg["To"] = "mario@cliente.it"
        msg["Date"] = "Mon, 03 Mar 2025 09:15:00 +0100"
        msg.set_content(body)
        msg.add_attachment(
            b"%PDF-1.4", maintype="application", subtype="pdf", filename="offerta.pdf"
        )
        return bytes(msg)

    def test_headers_body_and_attachments(self) -> None:
        raw = self._message(
            "Buongiorno,\n\nin allegato l'offerta aggiornata.\n\n"
            "Il giorno 01/03/2025 Mario ha scritto:\n> Potete mandarci l'offerta?"
        )
        parsed = EmailParser({"labels": "it"}).parse(_doc("m.eml", raw, "message/rfc822"), CTX)
        assert check_parsed_document(parsed) == []
        assert parsed.blocks[0].text == "Re: Offerta 2025"
        assert "Da: Anna Bianchi <anna@azienda.it>" in parsed.text
        assert "Allegati: offerta.pdf" in parsed.text
        # The quoted request is the previous message's text, not this one's.
        assert "Potete mandarci" not in parsed.text
        assert parsed.metadata["quoted_removed"] is True
        assert parsed.metadata["sent_date"] == date(2025, 3, 3)
        assert parsed.metadata["attachment_names"] == ["offerta.pdf"]

    def test_a_bare_forward_keeps_what_it_forwards(self) -> None:
        body = "-----Messaggio originale-----\nDa: x@y.it\nOggetto: contratto\n\nTesto utile."
        own, removed = strip_quoted(body)
        assert "Testo utile." in own and removed is False


def _signed(content: bytes) -> bytes:
    from tests.test_scanner import _signed as sign

    return sign(content)


def _pec_with_invoice(invoice: bytes) -> bytes:
    import email
    import email.policy

    original = EmailMessage()
    original["Subject"] = "Invio fattura 2025/123"
    original["From"] = "fatture@rossiforniture.it"
    original["To"] = "amministrazione@bianchi.it"
    original["Date"] = "Sat, 15 Mar 2025 10:00:00 +0100"
    original.set_content("In allegato la fattura elettronica firmata.")
    original.add_attachment(
        _signed(invoice),
        maintype="application",
        subtype="pkcs7-mime",
        filename="IT01234567890_00001.xml.p7m",
    )
    pec = EmailMessage()
    pec["Subject"] = "POSTA CERTIFICATA: Invio fattura 2025/123"
    pec["From"] = "per conto di: fatture@rossiforniture.it <posta-certificata@pec.gestore.it>"
    pec["To"] = "amministrazione@pec.bianchi.it"
    pec["Date"] = "Sat, 15 Mar 2025 10:00:05 +0100"
    pec.set_content("Messaggio di posta certificata.")
    pec.add_attachment(
        email.message_from_bytes(bytes(original), policy=email.policy.default)
    )  # a message/rfc822 part, as a PEC carries its original
    # The attached message needs the name every PEC gives it.
    last = list(pec.iter_attachments())[-1]
    last.set_param("filename", "postacert.eml", header="Content-Disposition")
    pec.add_attachment(
        b"<postacert tipo='posta-certificata'><dati>"
        b"<identificativo>opec-1</identificativo></dati></postacert>",
        maintype="application",
        subtype="xml",
        filename="daticert.xml",
    )
    return bytes(pec)


CONFIG = """
schema_version: 1
project: {{name: archivio}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params: {{root: {data}, expand: [eml, p7m]}}
ingestion:
  parse:
    enabled: true
    routes:
      - {{when: {{media_type: message/rfc822}}, impl: email, params: {{labels: it}}}}
      - {{when: {{media_type: application/xml}}, impl: fatturapa}}
    default: {{impl: text}}
  segment: {{impl: structural, max_tokens: 400}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: rules
    params:
      field_lexicon: [importo_totale, data_documento, cedente_denominazione, tipo_documento]
      field_types: {{importo_totale: float, data_documento: date, cedente_denominazione: str,
                     tipo_documento: str}}
      field_aliases: {{importo_totale: [importo, totale], data_documento: [data]}}
      value_aliases: {{tipo_documento: {{fattura: [fatture]}}}}
      default_date_field: data_documento
      default_measure: importo_totale
      locale: it
      level: document
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
"""


class TestArchive:
    def test_a_signed_invoice_inside_a_pec_answers_an_italian_question(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("cryptography")
        data = tmp_path / "archivio"
        data.mkdir()
        (data / "pec-2025-03-15.eml").write_bytes(_pec_with_invoice(_invoice()))
        (data / "pec-small.eml").write_bytes(
            _pec_with_invoice(_invoice("122.00").replace(b"2025/123", b"2025/124"))
        )
        cfg = tmp_path / "c.yaml"
        cfg.write_text(CONFIG.format(root=tmp_path.as_posix(), data=data.as_posix()))
        a = assemble(cfg)
        res = a.ingestion().build()
        assert res.ok, res.failures

        resp = a.query_engine().query("fatture con importo superiore a 1.000 euro")
        assert str(resp.decision.path) == RoutePath.STRUCTURED, resp.decision.reason
        assert resp.records is not None and not resp.records.is_empty()
        docs = {r["document_id"] for r in resp.records.as_dicts()}
        cards: dict[str, Any] = a.indexes["fields"].document_fields(docs)  # type: ignore[attr-defined]
        assert [c["numero_documento"] for c in cards.values()] == ["2025/123"]
        card = next(iter(cards.values()))
        # The envelope's facts travel with the invoice it carried.
        assert card["signers"] == "Mario Rossi"
        assert card["email_subject"] == "Invio fattura 2025/123"
        assert card["pec_identificativo"] == "opec-1"

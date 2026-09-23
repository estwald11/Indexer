"""FatturaPA: the Italian e-invoice, read as the structured record it already is.

Every invoice an Italian company sends or receives through the Sistema di
Interscambio is an XML file with the numbers already typed: document type,
number, date, total, taxable amount and VAT per rate, supplier and customer
with their VAT numbers, due dates, IBAN. Running an extractor -- regex or LLM
-- over a rendering of it would turn exact values back into guesses. So this
parser reads the XML and puts those values into document metadata, where they
reach every index as filterable fields and the structured index as columns;
it also renders the invoice as readable Italian text, with the lines and the
VAT summary as tables, for the retrieval surfaces.

Handles FPA12/FPR12 (``FatturaElettronica`` v1.2.x), any namespace prefix,
files carrying several invoices (a *lotto*: the first body's facts become the
document's, and the count is recorded), and ``.xml.p7m`` once the scanner has
unwrapped the envelope.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from indexer.core.document import BlockKind, ParsedDocument, SourceDocument
from indexer.core.registry import register
from indexer.core.stages import StageContext
from indexer.impls.parse import _Builder
from indexer.impls.parse_office import render_table
from indexer.normalize import parse_date, parse_number
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["FatturaPAParser", "is_fatturapa", "read_fatturapa"]

#: TipoDocumento codes, as the SdI specification names them.
TIPI_DOCUMENTO = {
    "TD01": "fattura",
    "TD02": "acconto/anticipo su fattura",
    "TD03": "acconto/anticipo su parcella",
    "TD04": "nota di credito",
    "TD05": "nota di debito",
    "TD06": "parcella",
    "TD16": "integrazione fattura reverse charge interno",
    "TD17": "integrazione/autofattura per acquisto servizi dall'estero",
    "TD18": "integrazione per acquisto di beni intracomunitari",
    "TD19": "integrazione/autofattura per acquisto di beni ex art.17 c.2 DPR 633/72",
    "TD20": "autofattura per regolarizzazione e integrazione delle fatture",
    "TD21": "autofattura per splafonamento",
    "TD22": "estrazione beni da deposito IVA",
    "TD23": "estrazione beni da deposito IVA con versamento dell'IVA",
    "TD24": "fattura differita",
    "TD25": "fattura differita (art.21 c.4 lett. b)",
    "TD26": "cessione di beni ammortizzabili e passaggi interni",
    "TD27": "fattura per autoconsumo o per cessioni gratuite senza rivalsa",
    "TD28": "acquisti da San Marino con IVA",
}
#: The normalised type a structured question filters on ("le fatture", "le
#: note di credito"), coarser than the SdI code.
_KIND = {"TD04": "nota_credito", "TD05": "nota_debito", "TD06": "parcella"}
MODALITA_PAGAMENTO = {
    "MP01": "contanti",
    "MP02": "assegno",
    "MP03": "assegno circolare",
    "MP05": "bonifico",
    "MP08": "carta di pagamento",
    "MP09": "RID",
    "MP12": "RIBA",
    "MP19": "SEPA Direct Debit",
    "MP20": "SEPA Direct Debit CORE",
    "MP21": "SEPA Direct Debit B2B",
    "MP23": "PagoPA",
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(el: ElementTree.Element | None, path: str) -> ElementTree.Element | None:
    """Namespace-blind path lookup: FatturaPA files use every prefix there is."""
    if el is None:
        return None
    cur: ElementTree.Element | None = el
    for step in path.split("/"):
        if cur is None:
            return None
        cur = next((c for c in cur if _local(c.tag) == step), None)
    return cur


def _findall(el: ElementTree.Element | None, name: str) -> list[ElementTree.Element]:
    return [c for c in el if _local(c.tag) == name] if el is not None else []


def _text(el: ElementTree.Element | None, path: str) -> str:
    found = _find(el, path)
    return (found.text or "").strip() if found is not None and found.text else ""


def _amount(el: ElementTree.Element | None, path: str) -> float | None:
    raw = _text(el, path)
    # xs:decimal: always a dot, never grouping.
    return parse_number(raw, "en") if raw else None


def is_fatturapa(raw: bytes) -> bool:
    head = raw[:2048].decode("utf-8", errors="ignore")
    return "FatturaElettronica" in head


@dataclass(frozen=True, slots=True)
class _Party:
    denominazione: str
    piva: str
    codice_fiscale: str
    indirizzo: str


def _party(el: ElementTree.Element | None) -> _Party:
    anag = _find(el, "DatiAnagrafici")
    name = _text(anag, "Anagrafica/Denominazione") or " ".join(
        x for x in (_text(anag, "Anagrafica/Nome"), _text(anag, "Anagrafica/Cognome")) if x
    )
    paese, codice = _text(anag, "IdFiscaleIVA/IdPaese"), _text(anag, "IdFiscaleIVA/IdCodice")
    sede = _find(el, "Sede")
    address = ", ".join(
        x
        for x in (
            _text(sede, "Indirizzo")
            + (" " + _text(sede, "NumeroCivico") if _text(sede, "NumeroCivico") else ""),
            " ".join(x for x in (_text(sede, "CAP"), _text(sede, "Comune")) if x),
            _text(sede, "Provincia"),
            _text(sede, "Nazione"),
        )
        if x.strip()
    )
    return _Party(name, f"{paese}{codice}" if codice else "", _text(anag, "CodiceFiscale"), address)


def read_fatturapa(raw: bytes) -> dict[str, Any]:
    """The invoice as data: parties, and one record per body."""
    root = ElementTree.fromstring(raw)
    if _local(root.tag) not in ("FatturaElettronica", "FatturaElettronicaSemplificata"):
        raise ValueError(f"not a FatturaPA document: root element {_local(root.tag)!r}")
    header = _find(root, "FatturaElettronicaHeader")
    bodies = []
    for body in _findall(root, "FatturaElettronicaBody"):
        general = _find(body, "DatiGenerali/DatiGeneraliDocumento")
        lines = [
            {
                "numero": _text(ln, "NumeroLinea"),
                "descrizione": _text(ln, "Descrizione"),
                "quantita": _text(ln, "Quantita"),
                "unita": _text(ln, "UnitaMisura"),
                "prezzo_unitario": _text(ln, "PrezzoUnitario"),
                "prezzo_totale": _text(ln, "PrezzoTotale"),
                "aliquota": _text(ln, "AliquotaIVA"),
            }
            for ln in _findall(_find(body, "DatiBeniServizi"), "DettaglioLinee")
        ]
        summary = [
            {
                "aliquota": _text(r, "AliquotaIVA"),
                "natura": _text(r, "Natura"),
                "imponibile": _amount(r, "ImponibileImporto"),
                "imposta": _amount(r, "Imposta"),
                "esigibilita": _text(r, "EsigibilitaIVA"),
            }
            for r in _findall(_find(body, "DatiBeniServizi"), "DatiRiepilogo")
        ]
        payments = [
            {
                "modalita": _text(p, "ModalitaPagamento"),
                "scadenza": parse_date(_text(p, "DataScadenzaPagamento") or ""),
                "importo": _amount(p, "ImportoPagamento"),
                "iban": _text(p, "IBAN"),
            }
            for dp in _findall(body, "DatiPagamento")
            for p in _findall(dp, "DettaglioPagamento")
        ]
        orders = [
            _text(o, "IdDocumento")
            for o in _findall(_find(body, "DatiGenerali"), "DatiOrdineAcquisto")
            if _text(o, "IdDocumento")
        ]
        contracts = [
            _text(o, "IdDocumento")
            for o in _findall(_find(body, "DatiGenerali"), "DatiContratto")
            if _text(o, "IdDocumento")
        ]
        bodies.append(
            {
                "tipo": _text(general, "TipoDocumento"),
                "divisa": _text(general, "Divisa"),
                "data": parse_date(_text(general, "Data") or ""),
                "numero": _text(general, "Numero"),
                "totale": _amount(general, "ImportoTotaleDocumento"),
                "causale": " ".join(
                    (c.text or "").strip() for c in _findall(general, "Causale") if c.text
                ),
                "linee": lines,
                "riepilogo": summary,
                "pagamenti": payments,
                "ordini": orders,
                "contratti": contracts,
            }
        )
    return {
        "cedente": _party(_find(header, "CedentePrestatore")),
        "cessionario": _party(_find(header, "CessionarioCommittente")),
        "bodies": bodies,
    }


@dataclass(frozen=True, slots=True)
class FatturaPAParams:
    #: Tolerance, in currency units, when checking the declared total against
    #: taxable amount plus VAT.
    total_tolerance: float = 0.01


#: The facts a FatturaPA carries into metadata, and so into every index's
#: fields, with their types -- what the router needs to read "fatture sopra i
#: 1.000 euro" as a comparison on a number.
FATTURAPA_FIELDS: dict[str, str] = {
    "formato": "str",
    "tipo_documento": "str",
    "tipo_documento_sdi": "str",
    "numero_documento": "str",
    "data_documento": "date",
    "divisa": "str",
    "importo_totale": "float",
    "imponibile": "float",
    "imposta": "float",
    "causale": "str",
    "cedente_denominazione": "str",
    "cedente_piva": "str",
    "cedente_codice_fiscale": "str",
    "cessionario_denominazione": "str",
    "cessionario_piva": "str",
    "cessionario_codice_fiscale": "str",
    "data_scadenza": "date",
    "modalita_pagamento": "str",
    "iban": "str",
    "ordini_acquisto": "str",
    "contratti": "str",
    "totali_coerenti": "bool",
    "fatture_nel_file": "int",
    "numeri_documento": "str",
}


@register(
    "parse",
    "fatturapa",
    version="1",
    params_model=dataclass_params(FatturaPAParams),
    summary=(
        "FatturaPA e-invoices (FPA12/FPR12): exact typed facts as document metadata, "
        "lines and VAT summary as tables. Standard library."
    ),
    declares_fields=lambda _p: dict(FATTURAPA_FIELDS),
)
def _make_fatturapa(params: dict[str, Any], **_: Any) -> FatturaPAParser:
    return FatturaPAParser(params)


class FatturaPAParser(StageImpl):
    STAGE, IMPL, VERSION = "parse", "fatturapa", "1"

    def can_parse(self, doc: SourceDocument) -> float:
        if doc.media_type in ("application/xml", "text/xml"):
            return 0.9
        return 0.0

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        data = read_fatturapa(doc.load())
        ced: _Party = data["cedente"]
        ces: _Party = data["cessionario"]
        b = _Builder(doc.document_id, doc.source_uri)
        for body in data["bodies"]:
            self._render_body(b, body, ced, ces)
        meta = dict(doc.metadata)
        meta.update(self._facts(data))
        return b.finish(doc.content_hash, metadata=meta)

    def _render_body(self, b: _Builder, body: dict[str, Any], ced: _Party, ces: _Party) -> None:
        tipo = TIPI_DOCUMENTO.get(body["tipo"], body["tipo"] or "documento")
        when = body["data"].strftime("%d/%m/%Y") if isinstance(body["data"], date) else ""
        b.add(
            f"{tipo.capitalize()} n. {body['numero']} del {when}".strip(),
            BlockKind.HEADING,
            level=1,
        )
        facts = [f"Tipo documento: {body['tipo']} ({tipo})"]
        if body["totale"] is not None:
            facts.append(f"Importo totale: {_euro(body['totale'])} {body['divisa'] or ''}".rstrip())
        if body["causale"]:
            facts.append(f"Causale: {body['causale']}")
        if body["ordini"]:
            facts.append(f"Ordini di acquisto: {', '.join(body['ordini'])}")
        if body["contratti"]:
            facts.append(f"Contratti: {', '.join(body['contratti'])}")
        b.add("\n".join(facts), BlockKind.PARAGRAPH)
        for title, party in (("Cedente/prestatore", ced), ("Cessionario/committente", ces)):
            b.add(title, BlockKind.HEADING, level=2)
            lines = [party.denominazione]
            if party.piva:
                lines.append(f"Partita IVA: {party.piva}")
            if party.codice_fiscale:
                lines.append(f"Codice fiscale: {party.codice_fiscale}")
            if party.indirizzo:
                lines.append(f"Sede: {party.indirizzo}")
            b.add("\n".join(x for x in lines if x), BlockKind.PARAGRAPH)
        if body["linee"]:
            b.add("Dettaglio linee", BlockKind.HEADING, level=2)
            rows = [
                [
                    "N.",
                    "Descrizione",
                    "Quantità",
                    "U.M.",
                    "Prezzo unitario",
                    "Prezzo totale",
                    "IVA %",
                ]
            ]
            rows += [
                [
                    ln["numero"],
                    ln["descrizione"],
                    ln["quantita"],
                    ln["unita"],
                    ln["prezzo_unitario"],
                    ln["prezzo_totale"],
                    ln["aliquota"],
                ]
                for ln in body["linee"]
            ]
            rendered, table = render_table(rows)
            b.add(rendered, BlockKind.TABLE, table=table)
        if body["riepilogo"]:
            b.add("Riepilogo IVA", BlockKind.HEADING, level=2)
            rows = [["Aliquota %", "Natura", "Imponibile", "Imposta", "Esigibilità"]]
            rows += [
                [
                    r["aliquota"],
                    r["natura"],
                    _euro(r["imponibile"]),
                    _euro(r["imposta"]),
                    r["esigibilita"],
                ]
                for r in body["riepilogo"]
            ]
            rendered, table = render_table(rows)
            b.add(rendered, BlockKind.TABLE, table=table)
        if body["pagamenti"]:
            b.add("Pagamento", BlockKind.HEADING, level=2)
            rows = [["Modalità", "Scadenza", "Importo", "IBAN"]]
            rows += [
                [
                    MODALITA_PAGAMENTO.get(p["modalita"], p["modalita"]),
                    p["scadenza"].strftime("%d/%m/%Y") if isinstance(p["scadenza"], date) else "",
                    _euro(p["importo"]),
                    p["iban"],
                ]
                for p in body["pagamenti"]
            ]
            rendered, table = render_table(rows)
            b.add(rendered, BlockKind.TABLE, table=table)

    def _facts(self, data: dict[str, Any]) -> dict[str, Any]:
        """Document-level fields, from the first body. Exact, typed, filterable."""
        bodies = data["bodies"]
        if not bodies:
            return {}
        first = bodies[0]
        ced: _Party = data["cedente"]
        ces: _Party = data["cessionario"]
        imponibile = _sum(r["imponibile"] for r in first["riepilogo"])
        imposta = _sum(r["imposta"] for r in first["riepilogo"])
        totale = first["totale"]
        if totale is None and imponibile is not None and imposta is not None:
            totale = round(imponibile + imposta, 2)
        due = sorted(p["scadenza"] for p in first["pagamenti"] if isinstance(p["scadenza"], date))
        facts: dict[str, Any] = {
            "formato": "FatturaPA",
            "tipo_documento": _KIND.get(first["tipo"], "fattura"),
            "tipo_documento_sdi": first["tipo"],
            "numero_documento": first["numero"],
            "data_documento": first["data"],
            "divisa": first["divisa"],
            "importo_totale": totale,
            "imponibile": imponibile,
            "imposta": imposta,
            "causale": first["causale"],
            "cedente_denominazione": ced.denominazione,
            "cedente_piva": ced.piva,
            "cedente_codice_fiscale": ced.codice_fiscale,
            "cessionario_denominazione": ces.denominazione,
            "cessionario_piva": ces.piva,
            "cessionario_codice_fiscale": ces.codice_fiscale,
            "data_scadenza": due[0] if due else None,
            "modalita_pagamento": sorted(
                {
                    MODALITA_PAGAMENTO.get(p["modalita"], p["modalita"])
                    for p in first["pagamenti"]
                    if p["modalita"]
                }
            ),
            "iban": sorted({p["iban"] for p in first["pagamenti"] if p["iban"]}),
            "ordini_acquisto": first["ordini"],
            "contratti": first["contratti"],
        }
        if totale is not None and imponibile is not None and imposta is not None:
            tol = float(self.param("total_tolerance", 0.01))
            # A declared total that does not add up is either a discount or
            # stamp duty the summary omits, or an error. Recorded either way.
            facts["totali_coerenti"] = abs(totale - (imponibile + imposta)) <= tol
        if len(bodies) > 1:
            facts["fatture_nel_file"] = len(bodies)
            facts["numeri_documento"] = [x["numero"] for x in bodies]
        return {k: v for k, v in facts.items() if v not in (None, "", [])}


def _sum(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals), 2) if vals else None


def _euro(value: float | None) -> str:
    """1234.5 -> '1.234,50': how the amount reads in an Italian document."""
    if value is None:
        return ""
    whole, frac = f"{value:,.2f}".split(".")
    return f"{whole.replace(',', '.')},{frac}"

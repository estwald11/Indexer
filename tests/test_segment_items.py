"""Segmenting a document by its own entries, and never by what it quotes.

``items`` makes one unit per numbered entry -- a specification's items, a price
list's lines, a procedure's steps -- however short, and segments a document
without enough entries as ``structural`` would. No segmenter makes a unit of
quoted text: the history below an email reply is in the document to be read,
not to be found.
"""

from __future__ import annotations

import re
from email.message import EmailMessage
from typing import Any

import pytest

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.document import BlockKind, ParsedDocument, SourceDocument
from indexer.core.ids import DocumentId, hash_bytes
from indexer.core.stages import StageContext
from indexer.core.unit import Unit
from indexer.eval.checks import check_parsed_document, check_units
from indexer.impls.parse import TextParser
from indexer.impls.parse_mail import EmailParser, split_quoted
from indexer.impls.parse_office import _paragraphs
from indexer.impls.segment import (
    DEFAULT_ITEM_PATTERN,
    ITEM_KIND,
    FixedWindowSegmenter,
    ItemSegmenter,
    StructuralSegmenter,
    WholeDocumentSegmenter,
)

CTX = StageContext(cache=NullCache(), accountant=InMemoryAccountant())

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


class _Doc(SourceDocument):
    __slots__ = ()

    def load(self) -> bytes:
        return self.metadata["body"]  # type: ignore[no-any-return]


def _source(name: str, body: bytes, media_type: str) -> SourceDocument:
    return _Doc(
        document_id=DocumentId(name),
        source_uri=f"mem://{name}",
        content_hash=hash_bytes(body),
        media_type=media_type,
        size_bytes=len(body),
        metadata={"body": body, "name": name},
    )


def _parse(body: str) -> ParsedDocument:
    return TextParser({}).parse(_source("d", body.encode(), "text/plain"), CTX)


def _segment(body: str, **params: Any) -> list[Unit]:
    parsed = _parse(body)
    units = list(ItemSegmenter(params).segment(parsed, CTX))
    assert check_units(units, parsed) == []
    return units


# --------------------------------------------------------------------------- #
# one unit per entry                                                           #
# --------------------------------------------------------------------------- #


class TestItems:
    def test_each_item_is_a_unit_however_short(self) -> None:
        units = _segment(CAPITOLATO)
        items = [u.text.split()[0] for u in units if u.kind == ITEM_KIND]
        assert items == [
            "03.01.003",
            "03.02.001",
            "03.02.002",
            "03.02.003",
            "03.03.001",
            "03.03.002",
        ]
        idem = next(u for u in units if u.text.startswith("03.02.002"))
        # Ten tokens, in the same paragraph as the item before it: its own unit.
        assert idem.text == "03.02.002 Idem c.s., ma per vuotatoi."

    def test_chapter_lines_are_the_section_path(self) -> None:
        units = _segment(CAPITOLATO)
        idem = next(u for u in units if u.text.startswith("03.02.002"))
        assert idem.section_path == ("03 IMPIANTO IDRICO-SANITARIO", "03.02 TELAI DI SOSTEGNO")
        pipe = next(u for u in units if u.text.startswith("03.03.002"))
        assert pipe.section_path == ("03 IMPIANTO IDRICO-SANITARIO", "03.03 TUBAZIONI")
        # The lines themselves are headings, not text of any unit.
        assert not any("TELAI DI SOSTEGNO" in u.text for u in units)
        assert units[0].kind != ITEM_KIND and units[0].text.startswith("Capitolato speciale")

    def test_an_item_runs_across_paragraphs_to_the_next_item(self) -> None:
        body = (
            "01.01.001 Collettore. Fornitura di collettore modulare in ottone.\n"
            "GEBERIT DUOFIX\n\n"
            "Compresa la cassetta da incasso con sportello.\n\n"
            "01.01.002 Idem, a sei vie.\n"
        )
        first, second = _segment(body, min_items=2)
        # A continuation on the next page, and a line in capitals inside an
        # item -- a brand, not a chapter -- both belong to the item.
        assert first.text.endswith("Compresa la cassetta da incasso con sportello.")
        assert "GEBERIT DUOFIX" in first.text and first.section_path == ()
        assert second.text == "01.01.002 Idem, a sei vie."

    @pytest.mark.parametrize(
        ("line", "opens"),
        [
            ("03.02.002 Idem c.s.", True),
            ("13.01.11.05*.a Vorwandelement", True),
            ("B.72.14.0013 Wie vor", True),
            ("13E.201.01 Kugelhahn", True),
            ("A.1C.4 Voce", True),
            ("Pos. 01.02.0030 Heizkörper", True),
            ("03.02.002", True),
            ("30.06.2025 revisione del documento", False),
            ("1.250.000 euro di lavori", False),
            ("12.50 m di tubo", False),
            ("DN 40/50 in PE", False),
        ],
    )
    def test_the_default_pattern_reads_codes_not_dates_or_amounts(
        self, line: str, opens: bool
    ) -> None:
        assert bool(re.match(DEFAULT_ITEM_PATTERN, line, re.M)) is opens

    def test_a_long_item_splits_and_its_later_pieces_say_what_they_belong_to(self) -> None:
        sentences = " ".join(f"Clausola {i} della fornitura del gruppo frigo." for i in range(80))
        units = _segment(
            f"05.01.001 Gruppo frigorifero. {sentences}\n", max_tokens=120, min_items=1
        )
        assert len(units) > 2 and all(u.kind == ITEM_KIND for u in units)
        assert units[0].section_path == ()
        assert all(u.section_path == ("05.01.001 Gruppo frigorifero.",) for u in units[1:])

    def test_a_document_without_enough_items_is_segmented_by_its_sections(self) -> None:
        # A letter that cites two items is not a list of items.
        letter = (
            "Spett.le Rossi Impianti,\n\n"
            "con riferimento alle voci\n03.02.001 e\n03.02.002 del capitolato, vi chiediamo "
            "un'offerta per la fornitura in opera entro fine mese.\n\n"
            "Cordiali saluti."
        )
        parsed = _parse(letter)
        params = {"max_tokens": 400, "merge_below_tokens": 24}
        items = ItemSegmenter(params).segment(parsed, CTX)
        assert items == StructuralSegmenter(params).segment(parsed, CTX)
        assert all(u.kind != ITEM_KIND for u in items)

    def test_a_pdf_can_keep_its_line_starts(self) -> None:
        page = "03.02.001 Telaio per\nlavabo.\n03.02.002 Idem c.s.,\nma per vuotatoi.\n\nNota."
        assert list(_paragraphs(page)) == [
            "03.02.001 Telaio per lavabo. 03.02.002 Idem c.s., ma per vuotatoi.",
            "Nota.",
        ]
        kept = list(_paragraphs(page, keep_line_breaks=True))
        assert kept[0].splitlines()[2] == "03.02.002 Idem c.s.,"


# --------------------------------------------------------------------------- #
# what a reply quotes is read, not found                                       #
# --------------------------------------------------------------------------- #

REPLY = """\
Buongiorno Anna,

va bene, procediamo con la seconda soluzione.

> Ci confermate anche il sopralluogo di giovedì?
Sì, giovedì alle 10.

Mario

Il giorno lun 3 mar 2025 alle 09:15 Anna Bianchi ha scritto:
> Vi proponiamo due soluzioni per la climatizzazione degli uffici di Bergamo:
> 1) sistema VRF con dodici unità interne, 48.500 euro;
> 2) pompa di calore aria-acqua da 14 kW con ventilconvettori, 36.900 euro.
"""


def _email(body: str, **params: Any) -> ParsedDocument:
    msg = EmailMessage()
    msg["Subject"] = "Re: Climatizzazione uffici di Bergamo"
    msg["From"] = "Mario Rossi <m.rossi@bl.it>"
    msg["To"] = "anna.bianchi@rossiimpianti.it"
    msg["Date"] = "Tue, 04 Mar 2025 10:00:00 +0100"
    msg.set_content(body)
    doc = _source("m.eml", bytes(msg), "message/rfc822")
    parsed = EmailParser({"labels": "it", **params}).parse(doc, CTX)
    assert check_parsed_document(parsed) == []
    return parsed


class TestQuoted:
    def test_a_reply_keeps_what_it_answers_as_quoted_context(self) -> None:
        parsed = _email(REPLY)
        quoted = [b.text for b in parsed.blocks if b.kind == BlockKind.QUOTED]
        # Answered inline, in place; the history, after the message's own text.
        assert quoted[0] == "> Ci confermate anche il sopralluogo di giovedì?"
        assert quoted[1].startswith("Il giorno lun 3 mar 2025 alle 09:15 Anna Bianchi")
        assert "pompa di calore aria-acqua da 14 kW" in quoted[1]
        assert parsed.blocks[-1].text == quoted[1]
        assert parsed.metadata["quoted_removed"] is True

    @pytest.mark.parametrize(
        "segmenter",
        [
            StructuralSegmenter({"max_tokens": 400}),
            ItemSegmenter({"max_tokens": 400}),
            FixedWindowSegmenter({"window_tokens": 40, "overlap_tokens": 8}),
            WholeDocumentSegmenter({}),
        ],
        ids=["structural", "items", "fixed_window", "whole_document"],
    )
    def test_no_segmenter_makes_a_unit_of_it(self, segmenter: Any) -> None:
        parsed = _email(REPLY)
        units = list(segmenter.segment(parsed, CTX))
        assert check_units(units, parsed) == []
        text = "\n".join(u.text for u in units)
        assert "procediamo con la seconda soluzione" in text
        assert "Sì, giovedì alle 10." in text
        assert "sopralluogo" not in text and "pompa di calore" not in text
        quoted = [b.provenance.span for b in parsed.blocks if b.kind == BlockKind.QUOTED]
        for u in units:
            s = u.provenance.span
            assert not any(s.start < q.end and q.start < s.end for q in quoted)

    def test_short_text_after_it_is_not_folded_across_it(self) -> None:
        body = (
            "Un paragrafo abbastanza lungo da stare da solo, con qualche parola in piu "
            "per superare la soglia di fusione dei segmenti corti.\n\n"
            "> una domanda citata\nOk."
        )
        parsed = _email(body)
        units = StructuralSegmenter({"max_tokens": 400, "merge_below_tokens": 32}).segment(
            parsed, CTX
        )
        assert check_units(units, parsed) == []
        assert not any("una domanda citata" in u.text for u in units)

    def test_it_can_be_dropped_or_bounded(self) -> None:
        dropped = _email(REPLY, quoted_context=False)
        assert not any(b.kind == BlockKind.QUOTED for b in dropped.blocks)
        assert "pompa di calore" not in dropped.text
        bounded = _email(REPLY, max_quoted_chars=80)
        history = [b.text for b in bounded.blocks if b.kind == BlockKind.QUOTED][-1]
        assert len(history) <= 80 and history.startswith("Il giorno")

    def test_a_bare_forward_is_all_the_sender_s(self) -> None:
        body = "-----Messaggio originale-----\nDa: x@y.it\nOggetto: contratto\n\nTesto utile."
        runs, history = split_quoted(body)
        assert runs == [(body, False)] and history == ""


def test_a_document_that_quotes_nothing_is_one_whole_unit_as_before() -> None:
    parsed = _parse(CAPITOLATO)
    whole = WholeDocumentSegmenter({}).segment(parsed, CTX)
    assert [u.text for u in whole] == [parsed.text]

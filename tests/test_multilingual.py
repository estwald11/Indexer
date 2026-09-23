"""Non-English text must tokenize, segment and retrieve as well as English.

The tokenizer used an ASCII character class, so every accented or umlauted word
was cut into fragments: "Größe" became "gr" + "e", "città" became "citt". Text
still retrieved on small corpora because other words carried it, which is why
nothing failed -- the fragments just collided with unrelated words at scale.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from indexer.impls.segment import _split_long_block
from indexer.pipeline import assemble
from indexer.textutil import content_words, tokenize


class TestTokenizer:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Größe der Zonen", ["grösse", "der", "zonen"]),
            ("città più bella", ["città", "più", "bella"]),
            ("Übersicht, Öffnungszeiten", ["übersicht", "öffnungszeiten"]),
            ("perché è così", ["perché", "è", "così"]),
            ("requests.get and read-timeout", ["requests.get", "and", "read-timeout"]),
        ],
    )
    def test_words_stay_whole(self, text: str, expected: list[str]) -> None:
        assert tokenize(text) == expected

    def test_eszett_and_ss_fold_together(self) -> None:
        """German writes both; a query in one form must find the other."""
        assert tokenize("Straße") == tokenize("STRASSE") == tokenize("strasse")

    def test_composed_and_decomposed_accents_match(self) -> None:
        """Both forms occur in real files; without NFKC they never match."""
        composed = unicodedata.normalize("NFC", "città")
        decomposed = unicodedata.normalize("NFD", "città")
        assert composed != decomposed
        assert tokenize(composed) == tokenize(decomposed)

    def test_content_words_keep_accented_terms(self) -> None:
        assert "öffnungszeiten" in content_words("Die Öffnungszeiten des Büros")
        assert "università" in content_words("La università di Bolzano")


class TestSentenceSplit:
    def test_splits_before_non_ascii_capitals(self) -> None:
        """Every sentence here opens with a non-ASCII capital. The old ASCII
        lookahead found no boundary at all and returned one oversized piece."""
        text = " ".join(["Übersicht der Linien und Zonen im Netz."] * 30 + ["È tutto."])
        pieces = [text[a:b] for a, b in _split_long_block(text, max_tokens=60)]
        assert len(pieces) > 1
        assert all(p.startswith(("Übersicht", "È")) for p in pieces)
        assert "".join(pieces) == text  # offsets still cover the text exactly


DOCS = {
    "citta.md": (
        "# Guida alla città\n\nLa città di Bolzano è bilingue e ospita numerosi musei.\n\n"
        "## Trasporti pubblici\n\nGli autobus notturni partono dalla stazione ogni trenta "
        "minuti; è possibile pagare con la carta di credito.\n"
    ),
    "strasse.md": (
        "# Übersicht der Straßenbahn\n\nDie Straßenbahn fährt täglich zwischen dem "
        "Hauptbahnhof und der Universität.\n\n## Öffnungszeiten der Büros\n\nDas "
        "Kundenbüro ist von Montag bis Freitag geöffnet; samstags bleibt es geschlossen.\n"
    ),
    "grenze.md": (
        "# Grenzen und Regeln\n\nGrenzkontrollen finden an der Grenze statt. Eine "
        "Genehmigung ist erforderlich.\n"
    ),
    "tram.md": "# Tram timetable\n\nThe tram runs every ten minutes to the harbour.\n",
}


@pytest.fixture
def engine(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    for name, text in DOCS.items():
        (data / name).write_text(text, encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"extends: {Path('configs/reference.yaml').resolve()}\n"
        f"paths: {{store: {tmp_path}/i, cache: {tmp_path}/c, manifests: {tmp_path}/m, "
        f"artifacts: {tmp_path}/a}}\n"
        "corpus:\n  sources:\n    - impl: filesystem\n"
        f"      params: {{root: {data}, include: ['**/*.md']}}\n",
        encoding="utf-8",
    )
    a = assemble(cfg)
    res = a.ingestion().build()
    assert res.ok, res.failures
    return a.query_engine()


class TestEndToEnd:
    @pytest.mark.parametrize(
        ("query", "doc"),
        [
            ("città bilingue musei", "citta.md"),
            ("autobus notturni stazione", "citta.md"),
            ("Straßenbahn Hauptbahnhof Universität", "strasse.md"),
            ("Strassenbahn Hauptbahnhof", "strasse.md"),  # ss spelling finds ß
            ("Öffnungszeiten Kundenbüro samstags", "strasse.md"),
            ("Grenzkontrollen Genehmigung", "grenze.md"),
            ("tram harbour", "tram.md"),
        ],
    )
    def test_top_hit_is_the_right_document(self, engine, query: str, doc: str) -> None:
        resp = engine.query(query, top_k=3)
        assert resp.hits, query
        top = resp.hits[0].unit
        assert top is not None
        assert top.unit.metadata["name"] == doc, (
            query,
            [h.unit.unit.metadata["name"] for h in resp.hits if h.unit],
        )

    def test_citations_resolve_to_the_original_unicode_text(self, engine) -> None:
        """Offsets are in characters of the canonical text, not UTF-8 bytes."""
        resp = engine.query("Öffnungszeiten Kundenbüro", top_k=1)
        hit = resp.hits[0]
        source = Path(hit.provenance.source_uri.removeprefix("file://")).read_text(encoding="utf-8")
        assert hit.unit is not None
        assert "Kundenbüro" in hit.unit.unit.text
        assert hit.unit.unit.text.split("\n")[0].strip("# ") in source

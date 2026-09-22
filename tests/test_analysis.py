"""Text analysis: Unicode tokens, Italian stopwords and stems, and BM25 over them.

The tokenizer used to be ASCII-only. These tests pin both halves of the fix:
accented words survive whole, and ASCII input tokenizes exactly as before --
the second half is what keeps the English ablation and every hashed vector
bit-exact.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pytest

from indexer.analysis import Analyzer, fold_accents, italian_light_stem
from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.ids import DocumentId, hash_text, make_unit_id
from indexer.core.provenance import Provenance, Span
from indexer.core.stages import IndexQuery, StageContext
from indexer.core.unit import EnrichedUnit, Unit
from indexer.impls.index_lexical import BM25Index, LanguageBM25Index
from indexer.textutil import WORD_RE, content_words, detect_language, tokenize

CTX = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
_ASCII_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*")
_ASCII_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]{2,}")


class TestTokenizer:
    def test_accented_words_survive_whole(self) -> None:
        text = "La città è bella, perché l'attività della società è cresciuta"
        assert tokenize(text) == [
            "la",
            "città",
            "è",
            "bella",
            "perché",
            "l",
            "attività",
            "della",
            "società",
            "è",
            "cresciuta",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "Set the read-timeout for requests.get to 27 seconds (v2.31.0).",
            "snake_case_names, CamelCase, x86-64 and 1.0.0-rc1; e-mail: a@b.c",
            "  multiple   spaces\tand\nnewlines -- dashes -- and ... dots ",
        ],
    )
    def test_ascii_input_tokenizes_exactly_as_before(self, text: str) -> None:
        assert tokenize(text) == [m.group(0).lower() for m in _ASCII_TOKEN.finditer(text)]
        assert [m.group(0) for m in WORD_RE.finditer(text)] == [
            m.group(0) for m in _ASCII_WORD.finditer(text)
        ]

    def test_content_words_use_the_documents_language(self) -> None:
        text = "Le fatture della società sono state pagate dal fornitore"
        assert "della" in content_words(text)  # English list: the old behaviour
        it = content_words(text, language="it")
        assert "della" not in it and "fatture" in it and "fornitore" in it


class TestLanguageDetection:
    def test_detects_italian_and_english_prose(self) -> None:
        assert detect_language("Le fatture della società sono state pagate con bonifico") == "it"
        assert detect_language("The invoices of the company are paid with a transfer") == "en"

    def test_short_queries_are_undetermined_rather_than_guessed(self) -> None:
        assert detect_language("fatture 2024") == "und"
        assert detect_language("invoices paid") == "und"


class TestItalianLightStemmer:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("fattura", "fatture"),
            ("contratto", "contratti"),
            ("fornitore", "fornitori"),
            ("banchi", "banche"),
            ("pagamento", "pagamenti"),
            ("attività", "attivita"),
        ],
    )
    def test_number_and_gender_variants_meet(self, a: str, b: str) -> None:
        assert italian_light_stem(a) == italian_light_stem(b)

    def test_short_words_are_left_alone(self) -> None:
        # Stripping a vowel from five-letter words conflates unrelated ones.
        for w in ("caso", "casa", "fine", "nota", "note"):
            assert italian_light_stem(w) == w

    def test_accent_folding(self) -> None:
        assert fold_accents("perché perchè più città") == "perche perche piu citta"


class TestAnalyzer:
    def test_the_default_is_the_historical_tokenizer(self) -> None:
        a = Analyzer()
        text = "Le Fatture dell'anno, perché"
        assert a.is_identity
        assert a.analyze(text) == tokenize(text)

    def test_italian_analysis_removes_function_words_and_elisions(self) -> None:
        a = Analyzer(language="it", stemmer="light", stopwords=True, fold=True)
        assert a.analyze("Le fatture dell'attività, perché è cresciuta") == [
            "fattur",
            "attivit",
            "cresciut",
        ]

    def test_spelling_variants_of_accents_meet(self) -> None:
        a = Analyzer(language="it", stemmer="light", stopwords=True, fold=True)
        assert a.analyze_query("perche' piu' citta'") == a.analyze_query("perché più città")

    def test_auto_analyses_each_document_in_its_language(self) -> None:
        a = Analyzer(language="auto", stemmer="light", stopwords=True, fold=True)
        it = a.analyze("Le fatture della società sono state emesse con ritardo")
        en = a.analyze("The invoices of the company were issued with a delay")
        assert "fattur" in it and "della" not in it
        assert "invoices" in en and "the" not in en
        # A query too short to detect is analysed both ways.
        assert set(a.analyze_query("fatture emesse")) >= {"fattur", "emess"}

    def test_a_stemmer_without_a_language_is_a_config_error(self) -> None:
        with pytest.raises(ValueError, match="needs a language"):
            Analyzer(stemmer="light")

    def test_snowball_identity_names_its_backend(self) -> None:
        pytest.importorskip("snowballstemmer")
        a = Analyzer(language="it", stemmer="snowball")
        assert a.identity()["backend"]
        assert a.analyze("fatture fattura") == ["fattur", "fattur"]


def _unit(doc: str, text: str) -> EnrichedUnit:
    did = DocumentId(doc)
    return EnrichedUnit(
        unit=Unit(
            unit_id=make_unit_id(did, hash_text(text)),
            document_id=did,
            text=text,
            provenance=Provenance(document_id=did, span=Span(0, len(text))),
        )
    )


ITALIAN = {
    "a": "La fattura numero 12 è stata emessa dal fornitore Rossi.",
    "b": "Il contratto di fornitura prevede pagamenti trimestrali.",
    "c": "Le fatture del fornitore Bianchi sono state pagate in ritardo.",
    "d": "Verbale della riunione del consiglio di amministrazione.",
}


class TestLanguageBM25:
    def test_plural_query_finds_singular_documents(self) -> None:
        legacy = BM25Index({}, name="lexical")
        italian = LanguageBM25Index(
            {"language": "it", "stemmer": "light", "stopwords": True, "fold_accents": True}
        )
        units = [_unit(k, v) for k, v in ITALIAN.items()]
        legacy.upsert(units, CTX)
        italian.upsert(units, CTX)

        q = IndexQuery(text="fatture emesse dai fornitori", top_k=10)
        legacy_docs = {h.document_id for h in legacy.search(q, CTX).hits}
        italian_docs = {h.document_id for h in italian.search(q, CTX).hits}
        # Word forms as written: "fatture" only matches c, and "fornitori"
        # matches nothing. Stemmed: both invoices and both suppliers are found.
        assert "a" not in legacy_docs
        assert {"a", "c"} <= italian_docs

    def test_scores_equal_a_brute_force_bm25(self) -> None:
        """The inverted index is an optimisation of the same arithmetic."""
        idx = LanguageBM25Index({"language": "it", "stemmer": "light", "stopwords": True})
        idx.upsert([_unit(k, v) for k, v in ITALIAN.items()], CTX)
        query = "fatture fornitore pagamenti ritardo"
        got = {h.document_id: h.score for h in idx.search(IndexQuery(text=query), CTX).hits}

        docs = {k: Counter(idx.analyzer.analyze(v)) for k, v in ITALIAN.items()}
        n, avg = len(docs), sum(sum(c.values()) for c in docs.values()) / len(docs)
        want: dict[str, float] = {}
        for term in dict.fromkeys(idx.analyzer.analyze_query(query)):
            df = sum(1 for c in docs.values() if term in c)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for k, c in docs.items():
                f = c.get(term, 0)
                if f:
                    denom = f + 1.2 * (1 - 0.75 + 0.75 * sum(c.values()) / avg)
                    want[k] = want.get(k, 0.0) + idf * f * 2.2 / denom
        assert got.keys() == want.keys()
        for k in want:
            assert got[k] == pytest.approx(want[k])

    def test_changing_the_analyser_rebuilds_the_stored_index(self, tmp_path: Path) -> None:
        path = tmp_path / "lex.json"
        plain = BM25Index({"path": str(path)})
        plain.upsert([_unit("a", ITALIAN["a"])], CTX)
        plain.flush()
        # Same file, stemming now on: the stored postings are unstemmed and
        # would never match a stemmed query term. The store starts empty and
        # the ledger restages every document for the fingerprint change.
        stemmed = BM25Index({"path": str(path), "language": "it", "stemmer": "light"})
        assert stemmed.stats().unit_count == 0
        assert stemmed.fingerprint() != plain.fingerprint()
        reopened = BM25Index({"path": str(path)})
        assert reopened.stats().unit_count == 1

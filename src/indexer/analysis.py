"""Text analysis for lexical matching: language, stopwords, accents, stems.

A leaf module, like ``indexer.textutil``, for the same reason: indexes, the
router, rerankers and the evaluation harness all need to agree on what a "term"
is, and none of them may import another's internals.

Why this exists at all. The frame's tokenizer was ASCII-only, and on an Italian
archive that is not a quality problem but a correctness one: "città" became
``citt``, "perché" became ``perch``, "è" vanished, and every accented word was
cut at its accent. On top of that, BM25 over raw word forms cannot match
"fatture" to "fattura" or "contratti" to "contratto" -- the plural/singular and
gender variation that Italian puts on nearly every noun and adjective -- and an
English-only stopword list leaves "della", "degli" and "che" as the most
"informative" terms in every document.

``Analyzer`` is configured, not guessed, and its identity is part of every
index fingerprint: an index built with one analyser and queried with another
matches nothing, so changing it must rebuild, and it does.

Stemming choices
----------------
``light``     Savoy's light stemmer for Italian (the one Lucene ships as
              ``ItalianLightStemmer``): folds accents and strips the final
              inflectional vowel(s) from words of six or more letters. Pure
              Python, no dependency, conservative -- it conflates number and
              gender and little else, which is most of what matters for search.
``snowball``  The Snowball stemmer (Italian or English). More aggressive -- it
              removes derivational suffixes too -- via ``PyStemmer`` or
              ``snowballstemmer``, whichever is installed; the backend and its
              version are part of the analyser's identity.
``none``      Word forms as written. The historical behaviour, kept as the
              default so the published ablation reproduces exactly.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from indexer.textutil import STOPWORDS, STOPWORDS_IT, detect_language, tokenize

__all__ = [
    "Analyzer",
    "fold_accents",
    "italian_light_stem",
    "snowball_stemmer",
]

LANGUAGES = ("none", "en", "it", "auto")
STEMMERS = ("none", "light", "snowball")


def fold_accents(text: str) -> str:
    """Strip diacritics: "perché" -> "perche", "città" -> "citta".

    Italian is typed inconsistently -- "perchè", "perché", "perche'" all occur,
    and so do "piu'" and "più" -- so matching on folded forms is closer to what
    the writer meant than matching on bytes.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


_VOWEL_FOLD = str.maketrans(
    {
        "à": "a",
        "á": "a",
        "â": "a",
        "ä": "a",
        "è": "e",
        "é": "e",
        "ê": "e",
        "ë": "e",
        "ì": "i",
        "í": "i",
        "î": "i",
        "ï": "i",
        "ò": "o",
        "ó": "o",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "ú": "u",
        "û": "u",
        "ü": "u",
    }
)


def italian_light_stem(word: str) -> str:
    """Savoy's light Italian stemmer.

    Words shorter than six letters are returned unchanged: stripping a vowel
    from "caso" or "fine" conflates words that have nothing to do with each
    other. Longer words lose their final inflectional vowel, and the plural
    "-ie"/"-ii"/"-hi"/"-he" and "-ia"/"-io" endings lose two letters, so that
    "fattura"/"fatture", "contratto"/"contratti", "banca"/"banche" meet.
    """
    if len(word) < 6:
        return word
    w = word.translate(_VOWEL_FOLD)
    last, prev = w[-1], w[-2]
    if last == "e":
        return w[:-2] if prev in "ih" else w[:-1]
    if last == "i":
        return w[:-2] if prev in "hi" else w[:-1]
    if last in "ao":
        return w[:-2] if prev == "i" else w[:-1]
    return w


@cache
def snowball_stemmer(language: str) -> tuple[Callable[[str], str], str]:
    """A Snowball stemmer for ``language`` and a string naming its backend.

    PyStemmer (C) is preferred for speed; ``snowballstemmer`` (pure Python) is
    the fallback. Both are generated from the same Snowball sources. The
    backend and version are returned so the analyser's identity -- and so the
    index fingerprint -- changes if the environment does.
    """
    from importlib.metadata import PackageNotFoundError, version

    name = {"it": "italian", "en": "english"}.get(language, language)
    try:
        import Stemmer

        s = Stemmer.Stemmer(name)
        try:
            v = version("PyStemmer")
        except PackageNotFoundError:  # pragma: no cover - vendored installs
            v = "?"
        return s.stemWord, f"pystemmer-{v}"
    except ImportError:
        pass
    try:
        import snowballstemmer
    except ImportError as exc:
        raise RuntimeError(
            "stemmer: snowball needs PyStemmer or snowballstemmer: "
            "pip install 'indexer[analysis]'. Use stemmer: light for a "
            "dependency-free Italian stemmer."
        ) from exc
    s2 = snowballstemmer.stemmer(name)
    try:
        v = version("snowballstemmer")
    except PackageNotFoundError:  # pragma: no cover
        v = "?"
    return s2.stemWord, f"snowballstemmer-{v}"


@dataclass(frozen=True)
class Analyzer:
    """Text -> terms, identically for documents and queries.

    ``language="auto"`` analyses each document in the language it is detected
    to be written in (Italian or English), and a query -- too short to detect
    reliably -- in every candidate language, searching the union of the terms.
    An archive of an Italian company holds English contracts too, and neither
    half should be analysed as the other.
    """

    language: str = "none"
    stemmer: str = "none"
    stopwords: bool = False
    fold: bool = False
    min_token_length: int = 1

    def __post_init__(self) -> None:
        if self.language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}, not {self.language!r}")
        if self.stemmer not in STEMMERS:
            raise ValueError(f"stemmer must be one of {STEMMERS}, not {self.stemmer!r}")
        if self.stemmer != "none" and self.language == "none":
            raise ValueError("a stemmer needs a language: set language to it, en or auto")

    # ------------------------------------------------------------------ api

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Analyzer:
        return cls(
            language=str(params.get("language", "none")),
            stemmer=str(params.get("stemmer", "none")),
            stopwords=bool(params.get("stopwords", False)),
            fold=bool(params.get("fold_accents", False)),
            min_token_length=int(params.get("min_token_length", 1)),
        )

    @property
    def is_identity(self) -> bool:
        """True when this analyser is exactly the historical ``tokenize``."""
        return (
            self.language == "none"
            and self.stemmer == "none"
            and not self.stopwords
            and not self.fold
            and self.min_token_length <= 1
        )

    def identity(self) -> dict[str, Any]:
        """What must match between the index and the query. Part of fingerprints."""
        out: dict[str, Any] = {
            "language": self.language,
            "stemmer": self.stemmer,
            "stopwords": self.stopwords,
            "fold": self.fold,
            "min_token_length": self.min_token_length,
        }
        if self.stemmer == "snowball":
            langs = ("it", "en") if self.language == "auto" else (self.language,)
            out["backend"] = sorted({snowball_stemmer(lang)[1] for lang in langs})
        return out

    def analyze(self, text: str) -> list[str]:
        """Terms of a document, in order (duplicates kept, for term frequency)."""
        if self.is_identity:
            return tokenize(text)
        lang = self._language_of(text)
        return list(self._terms(tokenize(text), lang))

    def analyze_query(self, text: str) -> list[str]:
        """Terms of a query. Distinct, and in every candidate language for auto."""
        if self.is_identity:
            return tokenize(text)
        tokens = tokenize(text)
        if self.language != "auto":
            return list(dict.fromkeys(self._terms(tokens, self.language)))
        detected = detect_language(text)
        langs = (detected,) if detected in ("it", "en") else ("it", "en")
        out: dict[str, None] = {}
        for lang in langs:
            out.update(dict.fromkeys(self._terms(tokens, lang)))
        return list(out)

    # ------------------------------------------------------------- internals

    def _language_of(self, text: str) -> str:
        if self.language != "auto":
            return self.language
        detected = detect_language(text)
        return detected if detected in ("it", "en") else "none"

    def _terms(self, tokens: Sequence[str], lang: str) -> Iterable[str]:
        stops = _stopwords_for(lang, folded=self.fold) if self.stopwords else frozenset()
        stem = self._stem_fn(lang)
        for t in tokens:
            if len(t) < self.min_token_length:
                continue
            # With folding on, "perche'" and "piu'" are the stopwords "perché"
            # and "più" typed without the accent, and must go the same way.
            if t in stops or (self.fold and fold_accents(t) in stops):
                continue
            if stem is not None:
                t = stem(t)
            if self.fold:
                t = fold_accents(t)
            if t:
                yield t

    def _stem_fn(self, lang: str) -> Callable[[str], str] | None:
        if self.stemmer == "none" or lang not in ("it", "en"):
            return None
        if self.stemmer == "light":
            # Savoy's light stemmer is defined for Italian. English gets its
            # Snowball stemmer only when asked for; "light" leaves it alone.
            return italian_light_stem if lang == "it" else None
        return snowball_stemmer(lang)[0]


@cache
def _stopwords_for(lang: str, *, folded: bool = False) -> frozenset[str]:
    base = {"it": STOPWORDS_IT, "en": STOPWORDS}.get(lang, frozenset())
    return base | {fold_accents(w) for w in base} if folded else base

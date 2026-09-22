"""Shared text primitives.

A leaf module, imported by implementations and by the evaluation harness alike.
It exists because the alternative was a cycle: the golden-set bootstrapper
needed the same tokenizer the lexical index uses, importing it from
``indexer.impls.index_lexical`` made ``indexer.eval`` depend on a particular
index implementation, and ``indexer.pipeline`` already depends on
``indexer.eval`` for the contract checks.

The layering rule this restores: **``indexer.eval`` must not depend on any
implementation.** The harness has to be able to evaluate a system this library
did not build -- a baseline someone wants to compare against -- and a harness
that imports a specific BM25 index cannot.

Nothing here is a contract. ``indexer.core`` stays free of it.
"""

from __future__ import annotations

import re

__all__ = [
    "STOPWORDS",
    "STOPWORDS_IT",
    "WORD_RE",
    "content_words",
    "detect_language",
    "hamming",
    "simhash64",
    "stopwords_for",
    "tokenize",
]

#: Word characters in any script, keeping dotted and hyphenated forms whole.
#: ``\w`` rather than ``[A-Za-z0-9_]``: the ASCII class cut every accented word
#: at its accent ("città" -> "citt", "perché" -> "perch") and dropped "è"
#: entirely. On ASCII input the two are identical, which is what keeps every
#: vector the hashing embedder produced for an English corpus bit-exact.
_TOKEN = re.compile(r"\w+(?:[.\-]\w+)*")

#: Matches identifier-ish words: at least three characters, starting with a
#: letter or underscore, dotted and hyphenated forms kept whole.
WORD_RE = re.compile(r"(?:[^\W\d]|_)[\w.\-]{2,}")

STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "is",
        "are",
        "be",
        "was",
        "were",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "by",
        "from",
        "not",
        "no",
        "if",
        "then",
        "than",
        "but",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "he",
        "she",
        "his",
        "her",
        "i",
        "me",
        "my",
        "us",
        "them",
        "use",
        "used",
        "using",
        "which",
        "what",
        "when",
        "where",
        "who",
        "how",
        "all",
        "any",
        "some",
        "each",
        "other",
        "more",
        "most",
        "such",
        "only",
        "own",
        "same",
        "so",
        "very",
        "s",
        "t",
        "just",
        "don",
        "now",
        "one",
        "two",
        "also",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "both",
        "few",
        "own",
        "too",
        "new",
        "see",
        "set",
        "get",
        "make",
        "like",
        "time",
        "way",
    ]
)


#: Italian function words, including the elided forms the tokenizer produces
#: when it splits at an apostrophe ("dell'attività" -> "dell", "attività").
#: Without them "della", "degli" and "che" are the most frequent -- and so, for
#: every heuristic that ranks terms by frequency, the most "distinctive" --
#: words of every Italian document.
STOPWORDS_IT = frozenset(
    """
    a ad al allo ai agli all alla alle agl anche anch ancora avere aveva avevano
    abbia abbiamo abbiano avete avrà avrebbe c che chi ci ciò coi col colla coll
    come con contro cui d da dal dallo dai dagli dall dalla dalle dei degli degl
    del dell della delle dello dentro di dopo dove dunque e è ed era erano essere
    fa fino fra gli ha hai hanno ho i il in invece io l la le lei li lo loro lui ma
    me mi mia mie miei mio molto ne negli nei nel nell nella nelle nello noi non
    nostra nostre nostri nostro o ogni oppure per perché perchè però più po poi
    presso qual quale quali qualche quando quanto quasi quel quell quella quelle
    quelli quello questa queste questi questo quest qui s se sé senza si sia siamo
    siano sono sopra sotto su sua sue sui sul sull sulla sulle sullo suo suoi
    ta tale tali te ti tra tu tua tue tuo tuoi tutti tutto tutte tutta un una uno
    vi voi vostra vostre vostri vostro sarà sarebbe essendo fosse fossero può
    possono deve devono ecc cioè ossia nonché
    """.split()  # noqa: SIM905 -- a word list reads better as prose than as 200 quoted items
)

_IT_MARKERS = frozenset(
    {"il", "della", "delle", "degli", "che", "per", "con", "sono", "una", "nel", "nella", "è"}
)
_EN_MARKERS = frozenset(
    {"the", "of", "and", "to", "is", "that", "with", "for", "are", "this", "be", "which"}
)


def stopwords_for(language: str) -> frozenset[str]:
    """The stopword list for ``it`` or ``en``; empty for anything else."""
    return {"it": STOPWORDS_IT, "en": STOPWORDS}.get(language, frozenset())


def detect_language(text: str, *, min_markers: int = 2) -> str:
    """ "it", "en" or "und", from function-word counts. Cheap and deterministic.

    Not a language identifier in general -- two languages, decided by the
    words no document can avoid -- because that is the decision the analysis
    needs, and a model for it would make every index fingerprint depend on a
    classifier's version. Returns "und" when the evidence is thin, which is
    the common case for a four-word query.
    """
    it = en = 0
    for m in _TOKEN.finditer(text[:20_000]):
        w = m.group(0).lower()
        if w in _IT_MARKERS:
            it += 1
        elif w in _EN_MARKERS:
            en += 1
    if max(it, en) < min_markers or it == en:
        return "und"
    return "it" if it > en else "en"


def tokenize(text: str) -> list[str]:
    """Lowercase word characters, keeping dotted and hyphenated identifiers whole.

    ``requests.get`` and ``read-timeout`` stay single tokens. In technical
    corpora those identifiers carry most of the discriminative signal, and
    splitting them turns a precise query into a common-word query.
    """
    return [m.group(0).lower() for m in _TOKEN.finditer(text)]


def simhash64(text: str, *, shingle: int = 3) -> int:
    """A 64-bit SimHash of word shingles: near-identical texts, near-identical bits.

    Two copies of a contract that differ by a date or a signature block land a
    few bits apart; unrelated texts land about 32 apart. Used to collapse near
    duplicates in results without a corpus-wide index -- comparing the few
    dozen hits of one query is all it takes.
    """
    import hashlib

    words = tokenize(text)
    grams = (
        [" ".join(words[i : i + shingle]) for i in range(len(words) - shingle + 1)]
        if len(words) >= shingle
        else [" ".join(words)]
    )
    weights = [0] * 64
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if h >> bit & 1 else -1
    return sum(1 << bit for bit in range(64) if weights[bit] > 0)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def content_words(text: str, *, min_length: int = 4, language: str = "en") -> list[str]:
    """Lowercased non-stopword terms, for "what is this about" heuristics."""
    stops = STOPWORDS | STOPWORDS_IT if language == "auto" else stopwords_for(language)
    return [
        w
        for m in WORD_RE.finditer(text)
        if (w := m.group(0).lower()) not in stops and not w.isdigit() and len(w) >= min_length
    ]

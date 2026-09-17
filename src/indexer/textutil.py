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

__all__ = ["STOPWORDS", "WORD_RE", "content_words", "tokenize"]

_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*")

#: Matches identifier-ish words: at least three characters, dotted and
#: hyphenated forms kept whole.
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]{2,}")

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


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumerics, keeping dotted and hyphenated identifiers whole.

    ``requests.get`` and ``read-timeout`` stay single tokens. In technical
    corpora those identifiers carry most of the discriminative signal, and
    splitting them turns a precise query into a common-word query.
    """
    return [m.group(0).lower() for m in _TOKEN.finditer(text)]


def content_words(text: str, *, min_length: int = 4) -> list[str]:
    """Lowercased non-stopword terms, for "what is this about" heuristics."""
    return [
        w
        for m in WORD_RE.finditer(text)
        if (w := m.group(0).lower()) not in STOPWORDS and not w.isdigit() and len(w) >= min_length
    ]

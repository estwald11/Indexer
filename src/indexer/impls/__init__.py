"""Reference implementations.

Importing this package registers every implementation below. That is the only
side effect, and it is why the assembler imports it once at startup rather than
each stage importing what it needs.

Third-party implementations register the same way, via the ``indexer.impls``
entry-point group, so an installed package is discovered without the frame
naming it.

Two implementations per stage, minimum, because a contract with one
implementation has not been tested as a contract -- it has been tested as a
description of that implementation.
"""

from indexer.impls import (
    corpus,
    enrich,
    fuse,
    index_dense,
    index_lexical,
    index_structured,
    parse,
    rerank,
    retrieve,
    route,
    segment,
)

__all__ = [
    "corpus",
    "enrich",
    "fuse",
    "index_dense",
    "index_lexical",
    "index_structured",
    "parse",
    "rerank",
    "retrieve",
    "route",
    "segment",
]

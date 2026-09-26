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
    enrich_entities,
    enrich_llm,
    enrich_resolve,
    fuse,
    index_dense,
    index_lexical,
    index_structured,
    parse,
    parse_external,
    parse_fatturapa,
    parse_mail,
    parse_office,
    rerank,
    retrieve,
    route,
    route_llm,
    segment,
)

__all__ = [
    "corpus",
    "enrich",
    "enrich_entities",
    "enrich_llm",
    "enrich_resolve",
    "fuse",
    "index_dense",
    "index_lexical",
    "index_structured",
    "parse",
    "parse_external",
    "parse_fatturapa",
    "parse_mail",
    "parse_office",
    "rerank",
    "retrieve",
    "route",
    "route_llm",
    "segment",
]

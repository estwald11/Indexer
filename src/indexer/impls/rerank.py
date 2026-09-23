"""Rerankers: noop, lexical-overlap, and a cross-encoder.

The contract is narrow on purpose: reorder and truncate the candidate set, never
add to it. A reranker that retrieves is a retriever, and letting that hide here
makes the latency budget unreadable -- the engine checks and raises.

Three implementations:

``noop``               Identity. Also the disabled behaviour, deliberately the
                       same arm so the ablation table cannot disagree with itself.
``lexical_overlap``    Offline. Query-term coverage plus proximity and a length
                       prior. Not a cross-encoder, and it is labelled as such.
``cross_encoder``      The real one, per the brief's BGE reranker guidance.
                       Needs a model download, so it cannot run in this
                       environment; the contract is what makes it a drop-in.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from indexer.core.query import Query
from indexer.core.registry import register
from indexer.core.results import RankedList
from indexer.core.stages import StageContext
from indexer.plugin import StageImpl, dataclass_params
from indexer.textutil import tokenize

__all__ = ["CrossEncoderReranker", "LexicalOverlapReranker", "NoopReranker"]


@dataclass(frozen=True, slots=True)
class NoopParams:
    pass


@register(
    "rerank",
    "noop",
    version="1",
    params_model=dataclass_params(NoopParams),
    summary="Identity. The same arm as rerank.enabled=false, by construction.",
)
def _make_noop(params: dict[str, Any], **_: Any) -> NoopReranker:
    return NoopReranker(params)


class NoopReranker(StageImpl):
    STAGE, IMPL, VERSION = "rerank", "noop", "1"

    def rerank(self, query: Query, candidates: RankedList, ctx: StageContext) -> RankedList:
        return candidates


@dataclass(frozen=True, slots=True)
class LexicalOverlapParams:
    #: Weight on the fraction of distinct query terms present. Coverage is the
    #: strongest single signal: a passage missing half the query's content words
    #: is usually answering a different question.
    coverage_weight: float = 0.55
    #: Weight on how tightly the matched terms cluster. Terms scattered across a
    #: long passage often co-occur by chance rather than by topic.
    proximity_weight: float = 0.25
    #: Weight on the prior rank, so fusion evidence is not discarded outright.
    prior_weight: float = 0.20
    #: Passages far from this length are mildly penalised: very short ones lack
    #: context, very long ones match everything.
    ideal_chars: int = 1200


@register(
    "rerank",
    "lexical_overlap",
    version="2",
    params_model=dataclass_params(LexicalOverlapParams),
    summary=(
        "Query-term coverage + proximity + prior rank. Offline stand-in for a "
        "cross-encoder; measurably weaker, and labelled as such."
    ),
)
def _make_lexical(params: dict[str, Any], **_: Any) -> LexicalOverlapReranker:
    return LexicalOverlapReranker(params)


class LexicalOverlapReranker(StageImpl):
    """An offline reranker with no model.

    What a cross-encoder does that this cannot: judge semantic relevance when
    the wording differs. What this does capture, and what carries a real part of
    reranking's benefit: term coverage and proximity, which first-stage scoring
    only approximates -- BM25 rewards a passage that repeats one query term
    many times over one that contains all of them once, and that is a large
    share of the top-20 errors reranking fixes.

    Expect a smaller effect than a cross-encoder's. The ablation report says so
    explicitly rather than letting the number stand in for one.
    """

    STAGE, IMPL, VERSION = "rerank", "lexical_overlap", "2"

    def rerank(self, query: Query, candidates: RankedList, ctx: StageContext) -> RankedList:
        if not candidates.hits:
            return candidates
        q_terms = [t for t in tokenize(query.text) if len(t) > 2]
        if not q_terms:
            return candidates
        q_set = set(q_terms)
        cw = float(self.param("coverage_weight", 0.55))
        pw = float(self.param("proximity_weight", 0.25))
        rw = float(self.param("prior_weight", 0.20))
        ideal = int(self.param("ideal_chars", 1200))
        n = len(candidates.hits)

        scored = []
        for h in candidates.hits:
            text = h.matched_text or (h.unit.indexing_text() if h.unit else "")
            toks = tokenize(text)
            tok_set = set(toks)
            coverage = len(q_set & tok_set) / len(q_set)
            proximity = _proximity(toks, q_set)
            prior = 1.0 - (h.rank - 1) / max(1, n)
            length_penalty = 1.0 / (1.0 + abs(math.log((len(text) + 1) / ideal)) * 0.15)
            score = (cw * coverage + pw * proximity + rw * prior) * length_penalty
            scored.append((h, score, coverage, proximity))

        ordered = sorted(scored, key=lambda t: (-t[1], t[0].unit_id))
        from dataclasses import replace

        return RankedList(
            hits=tuple(
                replace(
                    h,
                    rank=i,
                    score=s,
                    explain={
                        **dict(h.explain),
                        "coverage": round(cov, 3),
                        "proximity": round(prox, 3),
                        "prior_rank": h.rank,
                    },
                )
                for i, (h, s, cov, prox) in enumerate(ordered, start=1)
            ),
            source="rerank:lexical_overlap",
            query_text=query.text,
            fingerprint=self.fingerprint().key(),
        )


def _proximity(tokens: list[str], q_set: set[str]) -> float:
    """1.0 when all matched query terms sit in one tight window, → 0 when spread."""
    positions = [i for i, t in enumerate(tokens) if t in q_set]
    if len(positions) < 2:
        return 1.0 if positions else 0.0
    distinct = len({tokens[i] for i in positions})
    if distinct < 2:
        return 0.0
    # Smallest window containing the most distinct query terms.
    best = len(tokens)
    for i in range(len(positions)):
        seen: Counter[str] = Counter()
        for j in range(i, len(positions)):
            seen[tokens[positions[j]]] += 1
            if len(seen) == distinct:
                best = min(best, positions[j] - positions[i] + 1)
                break
    return distinct / max(distinct, best)


@dataclass(frozen=True, slots=True)
class CrossEncoderParams:
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    max_length: int = 512
    device: str = "cpu"


@register(
    "rerank",
    "cross_encoder",
    version="1",
    params_model=dataclass_params(CrossEncoderParams),
    summary=(
        "Cross-encoder (BGE family). The reference reranker; needs a model download. "
        "Bind to the simple path -- an LLM reranker there is too slow for most traffic."
    ),
    requires=("sentence-transformers",),
)
def _make_cross_encoder(params: dict[str, Any], **kw: Any) -> CrossEncoderReranker:
    return CrossEncoderReranker(params, model=kw.get("model"))


class CrossEncoderReranker(StageImpl):
    """The reference reranker.

    A cross-encoder scores (query, passage) jointly rather than comparing two
    independently-computed vectors, which is why it is both much better and much
    slower than first-stage retrieval -- and why it is a *reranker*, applied to
    a candidate set of tens rather than to the corpus.

    The brief's guidance holds in the config, not here: bind this to the simple
    path and keep an LLM reranker, if any, on the iterative path. The majority
    of traffic takes LOOKUP, and spending an LLM call there buys the least on
    the queries that need it least.
    """

    STAGE, IMPL, VERSION = "rerank", "cross_encoder", "1"

    def __init__(self, params: dict[str, Any], model: Any = None) -> None:
        super().__init__(params)
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "cross_encoder needs sentence-transformers: "
                    "pip install 'indexer[rerank-cross-encoder]'"
                ) from exc
            self._model = CrossEncoder(
                self.param("model"),
                max_length=int(self.param("max_length", 512)),
                device=self.param("device", "cpu"),
            )
        return self._model

    def rerank(self, query: Query, candidates: RankedList, ctx: StageContext) -> RankedList:
        if not candidates.hits:
            return candidates
        model = self._get_model()
        pairs = [
            (query.text, h.matched_text or (h.unit.indexing_text() if h.unit else ""))
            for h in candidates.hits
        ]
        scores = model.predict(pairs, batch_size=int(self.param("batch_size", 16)))
        from dataclasses import replace

        ordered = sorted(
            zip(candidates.hits, scores, strict=True), key=lambda t: (-float(t[1]), t[0].unit_id)
        )
        return RankedList(
            hits=tuple(
                replace(
                    h, rank=i, score=float(s), explain={**dict(h.explain), "prior_rank": h.rank}
                )
                for i, (h, s) in enumerate(ordered, start=1)
            ),
            source="rerank:cross_encoder",
            query_text=query.text,
            fingerprint=self.fingerprint().key(),
        )

"""Fusers: Reciprocal Rank Fusion and concatenation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from indexer.core.registry import register
from indexer.core.results import Hit, RankedList
from indexer.core.stages import StageContext
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["ConcatFuser", "RRFFuser"]


@dataclass(frozen=True, slots=True)
class RRFParams:
    k: int = 60
    weights: dict[str, float] = field(default_factory=dict)
    #: Per query type (factual, comparative, ...), weights that override
    #: ``weights`` for that type. Lexical evidence is worth more on a question
    #: quoting an article number than on a paraphrased one; one set of weights
    #: for both is a compromise the ablation can measure away.
    weights_by_type: dict[str, dict[str, float]] = field(default_factory=dict)


@register(
    "fuse",
    "rrf",
    version="1",
    params_model=dataclass_params(RRFParams),
    summary="Reciprocal Rank Fusion. Uses position only -- the one thing the lists share.",
)
def _make_rrf(params: dict[str, Any], **_: Any) -> RRFFuser:
    return RRFFuser(params)


class RRFFuser(StageImpl):
    """``score(u) = sum_i weight_i / (k + rank_i(u))``.

    Rank-based, not score-based, and that is the whole argument for it. A BM25
    score of 14.2 and a cosine similarity of 0.81 are not on a common scale, and
    any fuser that adds them is asserting a calibration it does not have. RRF
    discards the magnitudes and keeps the only comparable signal: position.

    ``k=60`` is the published default and a sane prior. It controls how sharply
    top ranks dominate; it is exposed because it is worth an ablation on a new
    corpus, not because the default is suspect.
    """

    STAGE, IMPL, VERSION = "fuse", "rrf", "1"

    def fuse(self, lists: Sequence[RankedList], ctx: StageContext) -> RankedList:
        k = int(self.param("k", 60))
        weights: dict[str, float] = dict(self.param("weights", {}) or {})
        qtype = (ctx.attrs or {}).get("query_type")
        by_type = self.param("weights_by_type", {}) or {}
        if qtype in by_type:
            weights.update(by_type[qtype])
        scores: dict[str, float] = {}
        best: dict[str, Hit] = {}
        contributors: dict[str, list[str]] = {}

        for rl in lists:
            w = float(weights.get(rl.source, 1.0))
            for h in rl.hits:
                scores[h.unit_id] = scores.get(h.unit_id, 0.0) + w / (k + h.rank)
                contributors.setdefault(h.unit_id, []).append(rl.source)
                # Keep the best-ranked occurrence, so the surviving hit carries
                # the provenance and text from the index that found it most
                # confidently.
                if h.unit_id not in best or h.rank < best[h.unit_id].rank:
                    best[h.unit_id] = h

        # Ties broken by unit id: fusion must be deterministic or every eval
        # delta becomes unreadable.
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        hits = tuple(
            replace(
                best[uid],
                rank=i,
                score=score,
                # Per-index contribution is what tells you whether the sparse
                # half is earning its place.
                explain={"rrf": score, "found_by": sorted(set(contributors[uid]))},
            )
            for i, (uid, score) in enumerate(ordered, start=1)
        )
        return RankedList(
            hits=hits,
            source="fuse:rrf",
            query_text=lists[0].query_text if lists else "",
            fingerprint=self.fingerprint().key(),
            total_candidates=len(scores),
        )


@dataclass(frozen=True, slots=True)
class ConcatParams:
    pass


@register(
    "fuse",
    "concat",
    version="1",
    params_model=dataclass_params(ConcatParams),
    summary="Concatenate in target order, dedup, keep first. The disabled-fuse arm.",
)
def _make_concat(params: dict[str, Any], **_: Any) -> ConcatFuser:
    return ConcatFuser(params)


class ConcatFuser(StageImpl):
    """Equivalent to any fuser for a single-index config, which is what makes
    "is hybrid worth it?" a clean two-arm comparison."""

    STAGE, IMPL, VERSION = "fuse", "concat", "1"

    def fuse(self, lists: Sequence[RankedList], ctx: StageContext) -> RankedList:
        seen: set[str] = set()
        hits: list[Hit] = []
        for rl in lists:
            for h in rl.hits:
                if h.unit_id in seen:
                    continue
                seen.add(h.unit_id)
                hits.append(h.with_rank(len(hits) + 1))
        return RankedList(
            hits=tuple(hits),
            source="fuse:concat",
            query_text=lists[0].query_text if lists else "",
            fingerprint=self.fingerprint().key(),
        )

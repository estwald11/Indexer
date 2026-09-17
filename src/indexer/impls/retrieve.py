"""Retrievers: sequential, parallel, and the bounded iterative loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from indexer.core.query import Query, RouteDecision, RouteTarget
from indexer.core.registry import register
from indexer.core.results import RankedList
from indexer.core.stages import Index, IndexQuery, StageContext
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["IterativeRetriever", "ParallelRetriever", "SequentialRetriever"]


def _search_one(
    idx: Index, target: RouteTarget, text: str, ctx: StageContext, step: int, on_error: str
) -> RankedList:
    try:
        rl = idx.search(IndexQuery(text=text, top_k=target.top_k, filters=target.filters), ctx)
    except Exception as exc:
        if on_error == "fail":
            raise
        # One index failing yields a shorter candidate set, not a failed query.
        # A reranker over three lists still works with two, and a query that
        # returns something beats a query that returns a stack trace.
        return RankedList(
            hits=(), source=target.index, query_text=text, fingerprint=f"error:{type(exc).__name__}"
        )
    return replace(rl, hits=tuple(replace(h, step=step) for h in rl.hits)) if step else rl


@dataclass(frozen=True, slots=True)
class SequentialParams:
    on_index_error: str = "degrade"


@register(
    "retrieve",
    "sequential",
    version="1",
    params_model=dataclass_params(SequentialParams),
    summary="Query each target in turn. The minimal implementation.",
)
def _make_sequential(params: dict[str, Any], **_: Any) -> SequentialRetriever:
    return SequentialRetriever(params)


class SequentialRetriever(StageImpl):
    STAGE, IMPL, VERSION = "retrieve", "sequential", "1"

    def retrieve(
        self,
        query: Query,
        decision: RouteDecision,
        indexes: Mapping[str, Index],
        ctx: StageContext,
    ) -> Sequence[RankedList]:
        on_error = self.param("on_index_error", "degrade")
        return [
            _search_one(indexes[t.index], t, query.text, ctx, 0, on_error)
            for t in decision.targets
            if t.index in indexes
        ]


@dataclass(frozen=True, slots=True)
class ParallelParams:
    max_workers: int = 8
    on_index_error: str = "degrade"


@register(
    "retrieve",
    "parallel",
    version="1",
    params_model=dataclass_params(ParallelParams),
    summary="Fan out across targets concurrently. Latency is the slowest index, not the sum.",
)
def _make_parallel(params: dict[str, Any], **_: Any) -> ParallelRetriever:
    return ParallelRetriever(params)


class ParallelRetriever(StageImpl):
    """Concurrent fan-out.

    Threads rather than async because the interesting index implementations are
    network calls or C extensions, both of which release the GIL, and requiring
    async would force every index implementation to have an async variant.
    """

    STAGE, IMPL, VERSION = "retrieve", "parallel", "1"

    def retrieve(
        self,
        query: Query,
        decision: RouteDecision,
        indexes: Mapping[str, Index],
        ctx: StageContext,
    ) -> Sequence[RankedList]:
        targets = [t for t in decision.targets if t.index in indexes]
        if len(targets) <= 1:
            return [
                _search_one(
                    indexes[t.index], t, query.text, ctx, 0, self.param("on_index_error", "degrade")
                )
                for t in targets
            ]
        on_error = self.param("on_index_error", "degrade")
        with ThreadPoolExecutor(max_workers=int(self.param("max_workers", 8))) as pool:
            futures = [
                pool.submit(_search_one, indexes[t.index], t, query.text, ctx, 0, on_error)
                for t in targets
            ]
            # Results in target order, not completion order: fusion weights are
            # keyed by index and a non-deterministic list order would make ties
            # resolve differently between runs.
            return [f.result() for f in futures]


@dataclass(frozen=True, slots=True)
class IterativeParams:
    max_workers: int = 8
    on_index_error: str = "degrade"
    #: Stop early when a round adds fewer than this many new units. Spending the
    #: whole budget when the candidate set has converged is pure latency.
    min_new_per_step: int = 3
    per_step_top_k: int = 25


@register(
    "retrieve",
    "iterative",
    version="1",
    params_model=dataclass_params(IterativeParams),
    summary="Bounded multi-step retrieval over the router's sub-queries. Honours step_budget.",
)
def _make_iterative(params: dict[str, Any], **_: Any) -> IterativeRetriever:
    return IterativeRetriever(params)


class IterativeRetriever(StageImpl):
    """The ITERATIVE path's loop -- inside a retriever, not a ninth stage.

    Keeping it here means all three route paths have the same pipeline shape, so
    no downstream stage has to know which path it is in. The loop is bounded by
    ``decision.step_budget``, which the engine verifies afterwards: the budget
    is a guarantee, not a hint.

    Step 0 is the original query. Later steps take the router's sub-queries.
    Each hit records its step, so the trace shows what each round contributed
    and whether the extra latency bought anything.
    """

    STAGE, IMPL, VERSION = "retrieve", "iterative", "1"

    def retrieve(
        self,
        query: Query,
        decision: RouteDecision,
        indexes: Mapping[str, Index],
        ctx: StageContext,
    ) -> Sequence[RankedList]:
        on_error = self.param("on_index_error", "degrade")
        per_step_k = int(self.param("per_step_top_k", 25))
        min_new = int(self.param("min_new_per_step", 3))
        targets = [t for t in decision.targets if t.index in indexes]

        queries = [query.text, *decision.sub_queries]
        # Deduplicate while preserving order: a decomposition that returns the
        # original query as its only part must not cost two identical rounds.
        queries = list(dict.fromkeys(queries))

        per_index: dict[str, list[Any]] = {t.index: [] for t in targets}
        seen_units: set[str] = set()

        for step in range(min(decision.step_budget, len(queries))):
            text = queries[step]
            new_this_step = 0
            for t in targets:
                rl = _search_one(
                    indexes[t.index],
                    replace(t, top_k=per_step_k if step else t.top_k),
                    text,
                    ctx,
                    step,
                    on_error,
                )
                for h in rl.hits:
                    if h.unit_id in seen_units:
                        continue
                    seen_units.add(h.unit_id)
                    per_index[t.index].append(h)
                    new_this_step += 1
            if step and new_this_step < min_new:
                break

        return [
            RankedList(
                hits=tuple(h.with_rank(i) for i, h in enumerate(hits, start=1)),
                source=name,
                query_text=query.text,
                fingerprint=self.fingerprint().key(),
            )
            for name, hits in per_index.items()
        ]

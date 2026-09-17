"""The query pipeline: route -> retrieve -> fuse -> rerank.

The engine is thin by design -- the stages do the work -- but it owns three
things no stage can own for itself:

**Invariant 5, enforced rather than trusted.** A ``STRUCTURED`` decision is
executed against structured indexes only. If the router returns that path with
a non-structured target, the engine raises rather than quietly querying a vector
index, because the quiet version is a correctness bug that shows up as "the
numbers are sometimes wrong" months later.

**The step budget.** ``LOOKUP`` gets one pass and the engine checks it. A
retriever that loops anyway is a contract violation, not a configuration choice.

**The trace.** Every decision is logged, including the ones made by default when
routing is off. Pre-fusion lists are kept on the response. Without them an
ablation between fuse implementations has no evidence, and a query that came
back wrong has no explanation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import CacheStore, NullCache
from indexer.core.errors import ContractViolation
from indexer.core.predicate import Predicate, all_of
from indexer.core.query import Query, RouteDecision, RoutePath, RouteTarget
from indexer.core.results import Hit, RankedList, RetrievalResponse
from indexer.core.stages import (
    Fuser,
    Index,
    Reranker,
    Retriever,
    Router,
    StageContext,
    StructuredCapable,
)
from indexer.pipeline.stores import UnitStore

__all__ = ["QueryEngine"]


class QueryEngine:
    """Executes a query against built indexes."""

    def __init__(
        self,
        *,
        indexes: Mapping[str, Index],
        router: Router,
        retriever: Retriever,
        fuser: Fuser,
        reranker: Reranker,
        unit_store: UnitStore | None = None,
        cache: CacheStore | None = None,
        route_enabled: bool = True,
        fuse_enabled: bool = True,
        rerank_enabled: bool = True,
        rerank_input_top_k: int = 50,
        rerank_output_top_k: int = 10,
        default_top_k: int = 50,
        decision_log: str | Path | None = None,
    ) -> None:
        self.indexes = dict(indexes)
        self.router = router
        self.retriever = retriever
        self.fuser = fuser
        self.reranker = reranker
        self.unit_store = unit_store
        self.cache = cache or NullCache()
        self.route_enabled = route_enabled
        self.fuse_enabled = fuse_enabled
        self.rerank_enabled = rerank_enabled
        self.rerank_input_top_k = rerank_input_top_k
        self.rerank_output_top_k = rerank_output_top_k
        self.default_top_k = default_top_k
        self.decision_log = Path(decision_log) if decision_log else None

    # ------------------------------------------------------------------ api

    def query(
        self, text: str, *, top_k: int = 20, filters: Predicate | None = None
    ) -> RetrievalResponse:
        return self.execute(Query(text=text, top_k=top_k, filters=filters))

    def execute(self, q: Query) -> RetrievalResponse:
        accountant = InMemoryAccountant()
        ctx = StageContext(cache=self.cache, accountant=accountant)
        latency: dict[str, float] = {}
        skipped: dict[str, str] = {}

        decision = self._route(q, ctx, latency, skipped)
        self._log_decision(q, decision)

        if str(decision.path) == RoutePath.STRUCTURED:
            return self._run_structured(q, decision, ctx, latency, skipped, accountant)

        t = time.perf_counter()
        targets = {t_.index for t_ in decision.targets}
        unknown = targets - set(self.indexes)
        if unknown:
            raise ContractViolation(
                f"router targeted indexes that are not configured: {sorted(unknown)}"
            )
        lists = list(self.retriever.retrieve(q, decision, self.indexes, ctx))
        latency["retrieve"] = (time.perf_counter() - t) * 1000

        steps_used = max((h.step for rl in lists for h in rl.hits), default=0) + 1
        if steps_used > decision.step_budget:
            raise ContractViolation(
                f"retriever used {steps_used} steps against a budget of "
                f"{decision.step_budget}; the budget is a guarantee, not a hint"
            )

        fused = self._fuse(lists, ctx, latency, skipped)
        reranked = self._rerank(q, fused, ctx, latency, skipped)

        final = reranked or fused
        hits = self._hydrate(final.hits[: q.top_k])
        return RetrievalResponse(
            query=q,
            decision=decision,
            hits=hits,
            retrieved=tuple(lists),
            fused=fused,
            reranked=reranked,
            latency_ms=latency,
            cost_usd=accountant.total_cost_usd(),
            skipped=skipped,
        )

    # -------------------------------------------------------------- stages

    def _route(
        self,
        q: Query,
        ctx: StageContext,
        latency: dict[str, float],
        skipped: dict[str, str],
    ) -> RouteDecision:
        t = time.perf_counter()
        if not self.route_enabled:
            skipped["route"] = "stage_disabled"
            latency["route"] = (time.perf_counter() - t) * 1000
            # Still a decision, still logged. An ablation with a hole in its
            # trace exactly where the comparison needs data is not an ablation.
            return RouteDecision(
                path=RoutePath.LOOKUP,
                targets=tuple(
                    RouteTarget(index=n, top_k=self.default_top_k, filters=q.filters)
                    for n in self.indexes
                ),
                step_budget=1,
                reason="stage_disabled",
                router="disabled",
            )
        decision = self.router.route(q, ctx)
        latency["route"] = (time.perf_counter() - t) * 1000

        # The caller's filters are always applied. It is not the router's place
        # to drop a tenant scope it did not understand.
        if q.filters is not None:
            decision = _with_caller_filters(decision, q.filters)
        return decision

    def _fuse(
        self,
        lists: Sequence[RankedList],
        ctx: StageContext,
        latency: dict[str, float],
        skipped: dict[str, str],
    ) -> RankedList:
        t = time.perf_counter()
        if not self.fuse_enabled:
            skipped["fuse"] = "stage_disabled"
            fused = _concat(lists)
        else:
            fused = self.fuser.fuse(lists, ctx)
        latency["fuse"] = (time.perf_counter() - t) * 1000
        return fused

    def _rerank(
        self,
        q: Query,
        fused: RankedList,
        ctx: StageContext,
        latency: dict[str, float],
        skipped: dict[str, str],
    ) -> RankedList | None:
        if not self.rerank_enabled:
            skipped["rerank"] = "stage_disabled"
            return None
        t = time.perf_counter()
        candidates = fused.top(self.rerank_input_top_k)
        out = self.reranker.rerank(q, candidates, ctx)

        # Reranking reorders and truncates; it never introduces a unit the
        # retrievers did not return. A reranker that retrieves is a retriever,
        # and hiding that here makes the latency budget unreadable.
        introduced = set(out.unit_ids()) - set(candidates.unit_ids())
        if introduced:
            raise ContractViolation(
                f"reranker introduced {len(introduced)} unit(s) not in the candidate set"
            )
        latency["rerank"] = (time.perf_counter() - t) * 1000
        return out.top(self.rerank_output_top_k)

    def _run_structured(
        self,
        q: Query,
        decision: RouteDecision,
        ctx: StageContext,
        latency: dict[str, float],
        skipped: dict[str, str],
        accountant: InMemoryAccountant,
    ) -> RetrievalResponse:
        """The structured path. Invariant 5 enforced, not trusted."""
        assert decision.structured_query is not None  # RouteDecision guarantees it
        targets = [t.index for t in decision.targets] or list(self.indexes)
        capable = [
            n
            for n in targets
            if n in self.indexes and isinstance(self.indexes[n], StructuredCapable)
        ]
        if not capable:
            raise ContractViolation(
                f"STRUCTURED route targeted {targets}, none of which answers structured "
                f"queries. Falling through to vector search here is exactly what "
                f"invariant 5 forbids, so this raises instead."
            )
        t = time.perf_counter()
        idx = self.indexes[capable[0]]
        assert isinstance(idx, StructuredCapable)
        records = idx.structured_query(decision.structured_query, ctx)
        latency["structured"] = (time.perf_counter() - t) * 1000
        skipped["retrieve"] = "structured_path"
        skipped["fuse"] = "structured_path"
        skipped["rerank"] = "structured_path"
        return RetrievalResponse(
            query=q,
            decision=decision,
            records=records,
            latency_ms=latency,
            cost_usd=accountant.total_cost_usd(),
            skipped=skipped,
        )

    # ------------------------------------------------------------ internals

    def _hydrate(self, hits: Sequence[Hit]) -> tuple[Hit, ...]:
        """Attach full units so a caller can render a passage and its citation.

        An index may legitimately store only ids and provenance; the unit store
        is where the text lives. Doing this at the end rather than per index is
        what keeps "add a fourth index" from duplicating the corpus a fourth time.
        """
        if self.unit_store is None:
            return tuple(hits)
        out: list[Hit] = []
        for h in hits:
            if h.unit is not None:
                out.append(h)
                continue
            eu = self.unit_store.get(h.unit_id)
            out.append(
                Hit(
                    unit_id=h.unit_id,
                    document_id=h.document_id,
                    rank=h.rank,
                    score=h.score,
                    index=h.index,
                    provenance=h.provenance,
                    unit=eu,
                    matched_text=h.matched_text or (eu.indexing_text() if eu else ""),
                    step=h.step,
                    explain=h.explain,
                )
            )
        return tuple(out)

    def _log_decision(self, q: Query, decision: RouteDecision) -> None:
        if self.decision_log is None:
            return
        rec = {"query": q.text, "ts": time.time(), **decision.as_log_record()}
        self.decision_log.parent.mkdir(parents=True, exist_ok=True)
        with self.decision_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _with_caller_filters(decision: RouteDecision, filters: Predicate) -> RouteDecision:
    from dataclasses import replace

    return replace(
        decision,
        targets=tuple(
            replace(t, filters=all_of([f for f in (t.filters, filters) if f is not None]))
            for t in decision.targets
        ),
    )


def _concat(lists: Sequence[RankedList]) -> RankedList:
    """Disabled-fuse behaviour: concatenate in order, dedup, keep first.

    Equivalent to any fuser for a single-index configuration, which is what
    makes "is hybrid worth it?" a clean two-arm comparison.
    """
    seen: set[str] = set()
    hits: list[Hit] = []
    for rl in lists:
        for h in rl.hits:
            if h.unit_id in seen:
                continue
            seen.add(h.unit_id)
            hits.append(h.with_rank(len(hits) + 1))
    return RankedList(hits=tuple(hits), source="fuse:disabled")

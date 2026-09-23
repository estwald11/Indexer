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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import CacheStore, NullCache
from indexer.core.errors import AccessDenied, ContractViolation
from indexer.core.ids import UnitId, hash_text
from indexer.core.predicate import Exists, In, Or, Predicate, StructuredQuery, all_of
from indexer.core.query import Query, RouteDecision, RoutePath, RouteTarget
from indexer.core.results import Hit, RankedList, RecordSet, RetrievalResponse
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
from indexer.textutil import hamming, simhash64

__all__ = ["AccessPolicy", "QueryEngine", "ShapePolicy"]


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    """Document-level access control. See ``config.schema.AccessConfig``."""

    enabled: bool = False
    field: str = "acl"
    missing: str = "deny"  # deny | allow

    def allows(self, fields: Mapping[str, Any], principals: Sequence[str] | None) -> bool:
        """Whether a unit or document with these fields is visible to
        ``principals`` -- the same rule the query path applies as a filter, for
        callers that fetch by id rather than search."""
        if not self.enabled:
            return True
        if principals is None:
            raise AccessDenied(
                "access control is on and the caller states no principals; pass the "
                "caller's user and groups"
            )
        acl = fields.get(self.field)
        if acl is None:
            return self.missing == "allow"
        granted = acl if isinstance(acl, list | tuple) else (acl,)
        return bool({str(g) for g in granted} & set(principals))


@dataclass(frozen=True, slots=True)
class ShapePolicy:
    """Final-list shaping for readers. See ``config.schema.ShapeConfig``."""

    enabled: bool = False
    max_per_document: int = 0
    collapse_duplicates: bool = True
    near_duplicate_bits: int = 0
    expand_neighbors: int = 0


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
        path_rerankers: Mapping[str, Reranker] | None = None,
        access: AccessPolicy | None = None,
        shape: ShapePolicy | None = None,
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
        #: Per-path overrides of the reranker, from ``route.paths.<p>.rerank``.
        #: This is where "a cross-encoder on LOOKUP, something slower on
        #: ITERATIVE" is expressed; the config key existed and was never read.
        self.path_rerankers = dict(path_rerankers or {})
        self.access = access or AccessPolicy()
        self.shape = shape or ShapePolicy()
        #: Most documents a ``document_filters`` scope searches.
        self.document_scope_limit = 20_000

    # ------------------------------------------------------------------ api

    def query(
        self,
        text: str,
        *,
        top_k: int = 20,
        filters: Predicate | None = None,
        principals: tuple[str, ...] | None = None,
        context: Sequence[str] = (),
        document_filters: Predicate | None = None,
    ) -> RetrievalResponse:
        """``context`` is the conversation so far, oldest first: a router that
        can read it makes a follow-up standalone before retrieval."""
        return self.execute(
            Query(
                text=text,
                top_k=top_k,
                filters=filters,
                principals=principals,
                context=tuple(context),
                document_filters=document_filters,
            )
        )

    def records(
        self,
        sq: StructuredQuery,
        *,
        filters: Predicate | None = None,
        principals: tuple[str, ...] | None = None,
    ) -> RecordSet:
        """A structured query as written, under the caller's scope and the
        access policy.

        For a caller that already knows the schema -- an agent that has read
        ``describe_schema`` -- and has no use for a router's reading of a
        sentence. The scope and the access filter apply exactly as on the
        structured path of ``execute``.
        """
        q = self._scoped(Query(text="", filters=filters, principals=principals))
        capable = [i for i in self.indexes.values() if isinstance(i, StructuredCapable)]
        if not capable:
            raise ContractViolation("no structured index is configured to answer records")
        if q.filters is not None:
            sq = replace(sq, where=all_of([w for w in (sq.where, q.filters) if w is not None]))
        ctx = StageContext(cache=self.cache, accountant=InMemoryAccountant())
        return capable[0].structured_query(sq, ctx)

    def execute(self, q: Query) -> RetrievalResponse:
        accountant = InMemoryAccountant()
        ctx = StageContext(cache=self.cache, accountant=accountant)
        latency: dict[str, float] = {}
        skipped: dict[str, str] = {}

        # Access control first, as a filter on the caller's scope: from here on
        # it is the caller's own filter, which every path already honours.
        q = self._scoped(q)
        q, scope = self._document_scope(q, ctx, skipped)
        decision = self._route(q, ctx, latency, skipped)
        if scope is not None:
            decision = replace(
                decision, targets=tuple(replace(t, unit_ids=scope) for t in decision.targets)
            )
        self._log_decision(q, decision)

        if str(decision.path) == RoutePath.STRUCTURED:
            return self._run_structured(q, decision, ctx, latency, skipped, accountant)

        # Retrieval asks the standalone form of a follow-up question, when the
        # router made one; the response keeps the question as asked.
        asked = replace(q, text=decision.rewritten_query) if decision.rewritten_query else q
        # What the fuser may condition on -- the question's type, the path --
        # travels in the context: fusion weights that suit a factual lookup
        # can be wrong for a comparison.
        ctx = replace(
            ctx,
            attrs={"query_type": str(decision.query_type), "route_path": str(decision.path)},
        )

        t = time.perf_counter()
        targets = {t_.index for t_ in decision.targets}
        unknown = targets - set(self.indexes)
        if unknown:
            raise ContractViolation(
                f"router targeted indexes that are not configured: {sorted(unknown)}"
            )
        lists = list(self.retriever.retrieve(asked, decision, self.indexes, ctx))
        latency["retrieve"] = (time.perf_counter() - t) * 1000

        steps_used = max((h.step for rl in lists for h in rl.hits), default=0) + 1
        if steps_used > decision.step_budget:
            raise ContractViolation(
                f"retriever used {steps_used} steps against a budget of "
                f"{decision.step_budget}; the budget is a guarantee, not a hint"
            )

        fused = self._fuse(lists, ctx, latency, skipped)
        reranked = self._rerank(asked, fused, ctx, latency, skipped, path=str(decision.path))

        final = reranked or fused
        if self.shape.enabled:
            t = time.perf_counter()
            hits = self._shape(self._hydrate(final.hits))[: q.top_k]
            latency["shape"] = (time.perf_counter() - t) * 1000
        else:
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

    # ------------------------------------------------------------- access

    def _scoped(self, q: Query) -> Query:
        """The query with the access policy conjoined into its filters."""
        if not self.access.enabled:
            return q
        if q.principals is None:
            raise AccessDenied(
                "access control is on and the query states no principals; pass the "
                "caller's user and groups (Query.principals)"
            )
        acl: Predicate = In(self.access.field, tuple(q.principals))
        if self.access.missing == "allow":
            acl = Or((acl, Exists(self.access.field, present=False)))
        return replace(q, filters=all_of([f for f in (q.filters, acl) if f is not None]))

    def _document_scope(
        self, q: Query, ctx: StageContext, skipped: dict[str, str]
    ) -> tuple[Query, tuple[UnitId, ...] | None]:
        """The units of the documents ``q.document_filters`` selects.

        Resolved over the structured index's document rows, where a document's
        fields are together; without one, the conditions fall back to each
        passage's own fields, and the response says so.
        """
        if q.document_filters is None:
            return q, None
        idx = next(
            (
                i
                for i in self.indexes.values()
                if isinstance(i, StructuredCapable) and hasattr(i, "unit_ids_for")
            ),
            None,
        )
        if idx is None:
            skipped["document_filters"] = "no structured index: applied to each passage"
            merged = all_of([w for w in (q.filters, q.document_filters) if w is not None])
            return replace(q, filters=merged, document_filters=None), None
        where = all_of([w for w in (q.document_filters, q.filters) if w is not None])
        rows = idx.structured_query(
            StructuredQuery(
                where=where,
                select=("document_id",),
                level="document",
                limit=self.document_scope_limit,
            ),
            ctx,
        )
        if rows.truncated:
            skipped["document_filters"] = (
                f"matched more than {self.document_scope_limit} documents; "
                f"searched the first {self.document_scope_limit}"
            )
        documents = [str(r[0]) for r in rows.rows]
        return q, tuple(idx.unit_ids_for(documents))

    # ------------------------------------------------------------- shaping

    def _shape(self, hits: Sequence[Hit]) -> tuple[Hit, ...]:
        """Diversify and dedupe the final list, and attach neighbouring text.

        Duplicates are *passages*, not documents: the same paragraph filed in
        three folders, or the privacy notice at the foot of every contract,
        is shown once with the others listed. Two versions of a contract keep
        the clauses that differ -- which is what a reader comparing them needs.
        """
        policy = self.shape
        kept: list[Hit] = []
        keys: list[tuple[str, int]] = []
        per_doc: dict[str, int] = {}
        for h in hits:
            text = h.unit.unit.text if h.unit else h.matched_text
            norm = " ".join(text.split()).casefold()
            key = (str(hash_text(norm)), simhash64(norm) if policy.near_duplicate_bits else 0)
            dup = self._duplicate_of(key, keys) if policy.collapse_duplicates else None
            if dup is not None:
                first = kept[dup]
                seen = list(first.explain.get("duplicates", []))
                seen.append({"document_id": h.document_id, "unit_id": h.unit_id})
                kept[dup] = replace(first, explain={**dict(first.explain), "duplicates": seen})
                continue
            if policy.max_per_document and per_doc.get(h.document_id, 0) >= policy.max_per_document:
                continue
            per_doc[h.document_id] = per_doc.get(h.document_id, 0) + 1
            kept.append(h)
            keys.append(key)
        out = [h.with_rank(i) for i, h in enumerate(kept, start=1)]
        if policy.expand_neighbors and self.unit_store is not None:
            out = [self._with_neighbours(h, policy.expand_neighbors) for h in out]
        return tuple(out)

    def _duplicate_of(self, key: tuple[str, int], keys: Sequence[tuple[str, int]]) -> int | None:
        bits = self.shape.near_duplicate_bits
        for i, (exact, sim) in enumerate(keys):
            if exact == key[0] or (bits and hamming(sim, key[1]) <= bits):
                return i
        return None

    def _with_neighbours(self, h: Hit, radius: int) -> Hit:
        assert self.unit_store is not None
        if h.unit is None:
            return h
        before: list[str] = []
        after: list[str] = []
        prev_id, next_id = h.unit.unit.prev_unit_id, h.unit.unit.next_unit_id
        for _ in range(radius):
            prev = self.unit_store.get(prev_id) if prev_id else None
            if prev is None:
                break
            before.insert(0, prev.unit.text)
            prev_id = prev.unit.prev_unit_id
        for _ in range(radius):
            nxt = self.unit_store.get(next_id) if next_id else None
            if nxt is None:
                break
            after.append(nxt.unit.text)
            next_id = nxt.unit.next_unit_id
        return replace(h, explain={**dict(h.explain), "before": before, "after": after})

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
        path: str = "",
    ) -> RankedList | None:
        if not self.rerank_enabled:
            skipped["rerank"] = "stage_disabled"
            return None
        t = time.perf_counter()
        candidates = fused.top(self.rerank_input_top_k)
        reranker = self.path_rerankers.get(path, self.reranker)
        out = reranker.rerank(q, candidates, ctx)

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
        # The caller's scope applies here exactly as it does to every retrieval
        # target. `_route` conjoins it onto targets, but the structured path
        # never reads target filters -- it runs `structured_query` -- so before
        # this line a tenant-scoped question over the structured index answered
        # from every tenant's records.
        sq = decision.structured_query
        extra = [w for w in (q.filters, q.document_filters) if w is not None]
        if extra:
            sq = replace(sq, where=all_of([w for w in (sq.where, *extra) if w is not None]))
        records = idx.structured_query(sq, ctx)
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

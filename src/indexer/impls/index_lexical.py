"""Lexical index: in-memory BM25.

Every index in this package indexes ``EnrichedUnit.indexing_text()`` and nothing
else as its retrieval surface. That is the contract, and it is where invariant
3's "before both embedding and lexical indexing" is actually honoured -- the
half most often forgotten is this one.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indexer.core.ids import DocumentId, UnitId
from indexer.core.predicate import Predicate, evaluate
from indexer.core.provenance import Provenance, Span
from indexer.core.registry import register
from indexer.core.results import Hit, RankedList
from indexer.core.stages import IndexQuery, IndexStatsView, IndexWriteReceipt, StageContext
from indexer.core.unit import EnrichedUnit
from indexer.io import atomic_write
from indexer.plugin import StageImpl, dataclass_params
from indexer.textutil import tokenize

__all__ = ["BM25Index"]


@dataclass(frozen=True, slots=True)
class BM25Params:
    k1: float = 1.2
    b: float = 0.75
    path: str = ""
    min_token_length: int = 1


@register(
    "index",
    "bm25_memory",
    version="2",
    params_model=dataclass_params(BM25Params),
    summary="In-memory BM25 over indexing_text(). The lexical baseline.",
)
def _make_bm25(params: dict[str, Any], **kw: Any) -> BM25Index:
    return BM25Index(params, name=kw.get("name", "lexical"))


class BM25Index(StageImpl):
    """Okapi BM25.

    Real, not a toy: BM25 is a strong baseline that hybrid systems frequently
    fail to beat on keyword-heavy corpora, which is exactly why invariant 4 says
    to fuse rather than replace. Keeping it in memory is a scale choice, not a
    quality one.
    """

    STAGE, IMPL, VERSION = "index", "bm25_memory", "2"
    kind = "lexical"

    def __init__(self, params: dict[str, Any], name: str = "lexical") -> None:
        super().__init__(params)
        self.name = name
        self.k1 = float(params.get("k1", 1.2))
        self.b = float(params.get("b", 0.75))
        self.path = Path(params["path"]) if params.get("path") else None
        self._docs: dict[str, dict[str, Any]] = {}
        self._df: Counter[str] = Counter()
        self._total_len = 0
        self._loaded = self.path is None
        self._dirty = False
        if self.path:
            self._load()

    # ------------------------------------------------------------- persistence

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text())
            self._docs = raw["docs"]
            self._df = Counter(raw["df"])
            self._total_len = raw["total_len"]

    def _save(self) -> None:
        if self.path:
            atomic_write(
                self.path,
                json.dumps(
                    {"docs": self._docs, "df": dict(self._df), "total_len": self._total_len}
                ).encode("utf-8"),
            )

    # ------------------------------------------------------------------ write

    def upsert(self, units: Sequence[EnrichedUnit], ctx: StageContext) -> IndexWriteReceipt:
        written = skipped = 0
        for eu in units:
            surface = eu.indexing_text()
            existing = self._docs.get(eu.unit_id)
            # Re-writing an identical unit must be a no-op, not a duplicate:
            # resumability depends on upsert being idempotent.
            if existing and existing["h"] == str(eu.indexing_hash):
                skipped += 1
                continue
            if existing:
                self._retract(eu.unit_id)
            tokens = tokenize(surface)
            tf = Counter(tokens)
            self._docs[eu.unit_id] = {
                "tf": dict(tf),
                "len": len(tokens),
                "h": str(eu.indexing_hash),
                "d": eu.document_id,
                "s": [eu.unit.provenance.span.start, eu.unit.provenance.span.end],
                "u": eu.unit.provenance.source_uri,
                "t": surface,
                "f": {k: _jsonable(v) for k, v in eu.fields().items()},
            }
            for term in tf:
                self._df[term] += 1
            self._total_len += len(tokens)
            written += 1
        self._dirty = True
        return IndexWriteReceipt(written=written, skipped=skipped)

    def _retract(self, unit_id: str) -> None:
        doc = self._docs.pop(unit_id, None)
        if not doc:
            return
        for term in doc["tf"]:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._total_len -= doc["len"]

    def delete(self, unit_ids: Sequence[UnitId], ctx: StageContext) -> int:
        n = 0
        for uid in unit_ids:
            if uid in self._docs:
                # Real removal, not a tombstone. A tombstoned index that does not
                # filter at query time keeps answering with deleted documents.
                self._retract(uid)
                n += 1
        self._dirty = True
        return n

    def delete_document(self, document_id: DocumentId, ctx: StageContext) -> int:
        ids = [uid for uid, d in self._docs.items() if d["d"] == document_id]
        return self.delete([UnitId(i) for i in ids], ctx)

    def flush(self) -> None:
        """Persist. Called once per build, not once per batch.

        Writing the whole file on every upsert is quadratic in corpus size --
        the reason this capability exists at all.
        """
        if self._dirty:
            self._save()
            self._dirty = False

    # ------------------------------------------------------------------- read

    def search(self, query: IndexQuery, ctx: StageContext) -> RankedList:
        if not self._docs:
            return RankedList(hits=(), source=self.name, query_text=query.text)
        terms = tokenize(query.text)
        n_docs = len(self._docs)
        avg_len = self._total_len / n_docs if n_docs else 1.0
        allowed = set(query.unit_ids) if query.unit_ids is not None else None

        scores: dict[str, float] = {}
        for term in set(terms):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for uid, doc in self._docs.items():
                f = doc["tf"].get(term)
                if not f:
                    continue
                if allowed is not None and uid not in allowed:
                    continue
                if query.filters is not None and not _passes(query.filters, doc["f"]):
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * doc["len"] / avg_len)
                scores[uid] = scores.get(uid, 0.0) + idf * (f * (self.k1 + 1)) / denom

        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: query.top_k]
        return RankedList(
            hits=tuple(
                self._hit(uid, rank, score) for rank, (uid, score) in enumerate(top, start=1)
            ),
            source=self.name,
            query_text=query.text,
            fingerprint=self.fingerprint().key(),
            total_candidates=len(scores),
        )

    def _hit(self, uid: str, rank: int, score: float) -> Hit:
        doc = self._docs[uid]
        return Hit(
            unit_id=UnitId(uid),
            document_id=DocumentId(doc["d"]),
            rank=rank,
            score=score,
            index=self.name,
            provenance=Provenance(
                document_id=DocumentId(doc["d"]),
                span=Span(doc["s"][0], doc["s"][1]),
                source_uri=doc["u"],
            ),
            matched_text=doc["t"],
        )

    def stats(self) -> IndexStatsView:
        return IndexStatsView(
            unit_count=len(self._docs),
            detail={
                "vocabulary": len(self._df),
                "mean_length": self._total_len / len(self._docs) if self._docs else 0,
                "k1": self.k1,
                "b": self.b,
            },
        )


def _jsonable(v: Any) -> Any:
    return v.isoformat() if hasattr(v, "isoformat") else v


def _passes(pred: Predicate, fields: dict[str, Any]) -> bool:
    try:
        return evaluate(pred, fields)
    except TypeError:
        # A type mismatch in a *filter* excludes rather than raises: the query
        # should return fewer results, not fail. The structured path raises,
        # because there the comparison is the answer.
        return False

"""Lexical index: in-memory BM25 over an inverted index.

Every index in this package indexes ``EnrichedUnit.indexing_text()`` and nothing
else as its retrieval surface. That is the contract, and it is where invariant
3's "before both embedding and lexical indexing" is actually honoured -- the
half most often forgotten is this one.

Two registrations share one implementation:

``bm25_memory``  The historical analyser -- word forms as written, no
                 stopwords, no stemming -- kept as the default so the published
                 ablation reproduces exactly.
``bm25``         Language-aware defaults: accents folded, stopwords removed,
                 Italian light stemming, each document analysed in the language
                 it is written in. What an Italian archive wants out of the box.

The analyser is part of the fingerprint and of the stored identity, so changing
it rebuilds the index rather than querying stemmed postings with unstemmed
terms.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indexer.analysis import Analyzer
from indexer.core.ids import DocumentId, UnitId
from indexer.core.predicate import Predicate, evaluate
from indexer.core.provenance import Provenance, Span
from indexer.core.registry import register
from indexer.core.results import Hit, RankedList
from indexer.core.stages import IndexQuery, IndexStatsView, IndexWriteReceipt, StageContext
from indexer.core.unit import EnrichedUnit
from indexer.io import atomic_write, decode_value, encode_value
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["BM25Index", "LanguageBM25Index"]


@dataclass(frozen=True, slots=True)
class BM25Params:
    k1: float = 1.2
    b: float = 0.75
    path: str = ""
    min_token_length: int = 1
    #: none | en | it | auto. ``auto`` analyses each unit in its detected
    #: language and a query in every candidate language.
    language: str = "none"
    #: none | light | snowball. ``light`` needs nothing; ``snowball`` needs the
    #: ``analysis`` extra.
    stemmer: str = "none"
    stopwords: bool = False
    fold_accents: bool = False
    #: With ``auto``: the language of a text none can be detected in.
    fallback_language: str = "none"


@dataclass(frozen=True, slots=True)
class LanguageBM25Params:
    k1: float = 1.2
    b: float = 0.75
    path: str = ""
    min_token_length: int = 1
    language: str = "auto"
    stemmer: str = "light"
    stopwords: bool = True
    fold_accents: bool = True
    #: The language of a document none can be detected in. ``it`` for an
    #: Italian archive; ``none`` leaves such a document unstemmed.
    fallback_language: str = "none"


@register(
    "index",
    "bm25_memory",
    version="3",
    params_model=dataclass_params(BM25Params),
    summary="In-memory BM25 over indexing_text(). The lexical baseline; word forms as written.",
)
def _make_bm25(params: dict[str, Any], **kw: Any) -> BM25Index:
    return BM25Index(params, name=kw.get("name", "lexical"))


@register(
    "index",
    "bm25",
    version="2",
    params_model=dataclass_params(LanguageBM25Params),
    summary=(
        "BM25 with language analysis: accents folded, stopwords removed, Italian "
        "light stemming, per-document language detection. For Italian archives."
    ),
)
def _make_language_bm25(params: dict[str, Any], **kw: Any) -> LanguageBM25Index:
    return LanguageBM25Index(params, name=kw.get("name", "lexical"))


class BM25Index(StageImpl):
    """Okapi BM25.

    Real, not a toy: BM25 is a strong baseline that hybrid systems frequently
    fail to beat on keyword-heavy corpora, which is exactly why invariant 4 says
    to fuse rather than replace. Keeping it in memory is a scale choice, not a
    quality one.

    Search walks the postings of the query's terms only. It used to scan every
    unit once per query term, which is invisible on a golden set and linear in
    the archive on every query -- the wrong shape for the path that is "paid
    forever".
    """

    STAGE, IMPL, VERSION = "index", "bm25_memory", "3"
    kind = "lexical"

    def __init__(self, params: dict[str, Any], name: str = "lexical") -> None:
        super().__init__(params)
        self.name = name
        self.k1 = float(params.get("k1", 1.2))
        self.b = float(params.get("b", 0.75))
        self.analyzer = Analyzer.from_params(params)
        self.path = Path(params["path"]) if params.get("path") else None
        self._docs: dict[str, dict[str, Any]] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._total_len = 0
        self._loaded = self.path is None
        self._dirty = False
        if self.path:
            self._load()

    def store_identity(self) -> dict[str, str]:
        # The analyser's resolved identity -- including which Snowball backend
        # and version -- not only its parameters: two machines with different
        # stemmer builds must not share postings.
        ident = super().store_identity()
        return {**ident, "analyzer": json.dumps(self.analyzer.identity(), sort_keys=True)}

    # ------------------------------------------------------------- persistence

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            stored = raw.get("identity")
            if stored is not None and stored != self.store_identity():
                # Built as something else -- another analyser, another version.
                # Start empty; the ledger restages every document for exactly
                # this change, and loading the old postings would make each
                # rewrite a no-op.
                self._dirty = True
                return
            self._docs = raw["docs"]
            for uid, d in self._docs.items():
                d["f"] = decode_value(d.get("f", {}))
                for term, f in d["tf"].items():
                    self._postings.setdefault(term, {})[uid] = f
            self._total_len = raw["total_len"]

    def _save(self) -> None:
        if self.path:
            docs = {uid: {**d, "f": encode_value(d["f"])} for uid, d in self._docs.items()}
            atomic_write(
                self.path,
                json.dumps(
                    {
                        "identity": self.store_identity(),
                        "docs": docs,
                        "total_len": self._total_len,
                    }
                ).encode("utf-8"),
            )

    # ------------------------------------------------------------------ write

    def upsert(self, units: Sequence[EnrichedUnit], ctx: StageContext) -> IndexWriteReceipt:
        written = skipped = 0
        for eu in units:
            existing = self._docs.get(eu.unit_id)
            # Re-writing an identical unit must be a no-op, not a duplicate:
            # resumability depends on upsert being idempotent. "Identical" is
            # the whole record -- span, fields, source -- not just the surface,
            # or a moved span and a corrected field are never written.
            record = str(eu.record_hash)
            if existing and existing.get("rh") == record:
                skipped += 1
                continue
            if existing:
                self._retract(eu.unit_id)
            surface = eu.indexing_text()
            # The document's language, detected on the whole document, not on
            # this unit: a one-line unit gives detection nothing to go on.
            doc_language = eu.unit.metadata.get("doc_language")
            tokens = self.analyzer.analyze(
                surface, doc_language if isinstance(doc_language, str) else None
            )
            tf = Counter(tokens)
            self._docs[eu.unit_id] = {
                "tf": dict(tf),
                "len": len(tokens),
                "h": str(eu.indexing_hash),
                "rh": record,
                "d": eu.document_id,
                "s": [eu.unit.provenance.span.start, eu.unit.provenance.span.end],
                "u": eu.unit.provenance.source_uri,
                "t": surface,
                # Typed in memory, tagged on disk. Written as ISO strings, a
                # date filter compared date against str and excluded everything.
                "f": dict(eu.filter_fields()),
            }
            for term, f in tf.items():
                self._postings.setdefault(term, {})[eu.unit_id] = f
            self._total_len += len(tokens)
            written += 1
        self._dirty = True
        return IndexWriteReceipt(written=written, skipped=skipped)

    def _retract(self, unit_id: str) -> None:
        doc = self._docs.pop(unit_id, None)
        if not doc:
            return
        for term in doc["tf"]:
            post = self._postings.get(term)
            if post is None:
                continue
            post.pop(unit_id, None)
            if not post:
                del self._postings[term]
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
        terms = list(dict.fromkeys(self.analyzer.analyze_query(query.text)))
        n_docs = len(self._docs)
        avg_len = self._total_len / n_docs if n_docs else 1.0
        allowed = set(query.unit_ids) if query.unit_ids is not None else None
        passes: dict[str, bool] = {}

        scores: dict[str, float] = {}
        for term in terms:
            post = self._postings.get(term)
            if not post:
                continue
            df = len(post)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for uid, f in post.items():
                if allowed is not None and uid not in allowed:
                    continue
                doc = self._docs[uid]
                if query.filters is not None:
                    ok = passes.get(uid)
                    if ok is None:
                        ok = passes[uid] = _passes(query.filters, doc["f"])
                    if not ok:
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
                "vocabulary": len(self._postings),
                "mean_length": self._total_len / len(self._docs) if self._docs else 0,
                "k1": self.k1,
                "b": self.b,
                "analyzer": self.analyzer.identity(),
            },
        )


class LanguageBM25Index(BM25Index):
    """BM25 with language-aware analysis on by default. See module docstring."""

    STAGE, IMPL, VERSION = "index", "bm25", "2"


def _passes(pred: Predicate, fields: dict[str, Any]) -> bool:
    try:
        return evaluate(pred, fields)
    except TypeError:
        # A type mismatch in a *filter* excludes rather than raises: the query
        # should return fewer results, not fail. The structured path raises,
        # because there the comparison is the answer.
        return False

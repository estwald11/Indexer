"""Dense index: deterministic offline embeddings, exact cosine search.

An honest framing of what this is. ``hash_embedding`` is a hashed
bag-of-n-grams projected into a fixed-dimensional space. It is a real vector
index -- exact cosine over dense vectors, with all the same write, delete and
filter semantics a production store needs -- but it is **not** a semantic
embedding model. It cannot match "how do I stop a request hanging" to "timeout"
unless the words overlap.

Why it is here anyway: the reference path must run offline, in CI, with no model
download and no credentials. That makes the pipeline testable and the ablation
reproducible. The cost is that **absolute** dense-retrieval numbers from this
index mean nothing, and only the deltas between arms are informative. Reading a
P@5 from this index as though it were a real embedding model's would be wrong.

The seam: a real dense index is this file with ``_embed`` replaced by a model
call, and nothing else in the frame changes. ``configs/full.yaml`` names Qdrant
with a Matryoshka model for exactly that reason.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indexer.core.ids import DocumentId, UnitId
from indexer.core.provenance import Provenance, Span
from indexer.core.registry import register
from indexer.core.results import Hit, RankedList
from indexer.core.stages import IndexQuery, IndexStatsView, IndexWriteReceipt, StageContext
from indexer.core.unit import EnrichedUnit
from indexer.impls.index_lexical import _jsonable, _passes
from indexer.textutil import tokenize
from indexer.io import atomic_write
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["HashEmbeddingIndex"]

# Optional accelerator. Exact brute-force cosine over ten thousand 384-dimension
# vectors is ~120ms per query in pure Python, which is fine for a unit test and
# painful for a 300-query ablation across eight arms. numpy makes it ~2ms and
# changes nothing about the results -- the same arithmetic, vectorised.
#
# Optional rather than required, because indexer.core promises a dependency-free
# frame and the reference path should run on a bare interpreter. The pure-Python
# path below is the definition; this is a faster way to compute the same thing,
# and a test asserts the two agree.
try:  # pragma: no cover - presence depends on the environment
    import numpy as _np

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    HAVE_NUMPY = False


@dataclass(frozen=True, slots=True)
class HashEmbeddingParams:
    dim: int = 256
    #: Character n-gram width. Character n-grams give some robustness to
    #: morphology ("timeout" / "timeouts") that word hashing alone lacks.
    char_ngram: int = 4
    use_words: bool = True
    use_char_ngrams: bool = True
    path: str = ""
    idf_weighting: bool = True


@register(
    "index",
    "hash_embedding",
    version="1",
    params_model=dataclass_params(HashEmbeddingParams),
    summary=(
        "Deterministic hashed embeddings, exact cosine. Offline and reproducible; "
        "NOT a semantic model -- absolute numbers are not meaningful, deltas are."
    ),
)
def _make_hash_embedding(params: dict[str, Any], **kw: Any) -> HashEmbeddingIndex:
    return HashEmbeddingIndex(params, name=kw.get("name", "dense"))


class HashEmbeddingIndex(StageImpl):
    STAGE, IMPL, VERSION = "index", "hash_embedding", "1"
    kind = "dense"

    def __init__(self, params: dict[str, Any], name: str = "dense") -> None:
        super().__init__(params)
        self.name = name
        self.dim = int(params.get("dim", 256))
        self.char_ngram = int(params.get("char_ngram", 4))
        self.use_words = bool(params.get("use_words", True))
        self.use_chars = bool(params.get("use_char_ngrams", True))
        self.idf_weighting = bool(params.get("idf_weighting", True))
        self.path = Path(params["path"]) if params.get("path") else None
        self._vecs: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._df: dict[str, int] = {}
        self._matrix: Any = None  # numpy cache, invalidated on write
        self._matrix_ids: list[str] = []
        self._loaded = self.path is None
        self._dirty = False
        if self.path:
            self._load()

    def _load(self) -> None:
        self._loaded = True
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text())
            self._vecs = raw["vecs"]
            self._meta = raw["meta"]
            self._df = raw.get("df", {})

    def _save(self) -> None:
        if self.path:
            atomic_write(
                self.path,
                json.dumps({"vecs": self._vecs, "meta": self._meta, "df": self._df}).encode(),
            )

    # -------------------------------------------------------------- embedding

    def _features(self, text: str) -> dict[str, float]:
        feats: dict[str, float] = {}
        if self.use_words:
            for w in tokenize(text):
                feats[f"w:{w}"] = feats.get(f"w:{w}", 0.0) + 1.0
        if self.use_chars:
            norm = re.sub(r"\s+", " ", text.lower())
            n = self.char_ngram
            for i in range(max(0, len(norm) - n + 1)):
                g = f"c:{norm[i : i + n]}"
                feats[g] = feats.get(g, 0.0) + 1.0
        return feats

    def _embed(
        self, text: str, *, use_idf: bool = True, feats: dict[str, float] | None = None
    ) -> list[float]:
        """Hash features into ``dim`` buckets with signed accumulation.

        The sign comes from a second hash bit, which keeps collisions from
        systematically inflating similarity -- the standard hashing-trick
        correction, and without it every long document looks similar to every
        other long document.
        """
        vec = [0.0] * self.dim
        n_docs = max(1, len(self._vecs))
        for feat, count in (feats if feats is not None else self._features(text)).items():
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            weight = 1.0 + math.log(count)
            if use_idf and self.idf_weighting:
                df = self._df.get(feat, 0)
                weight *= math.log(1 + n_docs / (1 + df))
            vec[bucket] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if not norm:
            return vec
        # Rounded to 5 places before storage. Cosine over unit vectors is
        # insensitive well below this, and full repr() floats cost roughly
        # three times the bytes for no measurable difference in ranking.
        return [round(v / norm, 5) for v in vec]

    # ------------------------------------------------------------------ write

    def upsert(self, units: Sequence[EnrichedUnit], ctx: StageContext) -> IndexWriteReceipt:
        written = skipped = 0
        for eu in units:
            if self._meta.get(eu.unit_id, {}).get("h") == str(eu.indexing_hash):
                skipped += 1
                continue
            surface = eu.indexing_text()
            feats = self._features(surface)
            for feat in feats:
                self._df[feat] = self._df.get(feat, 0) + 1
            # Documents stored unweighted, queries IDF-weighted at search time.
            # The dot product is then the standard TF-IDF scheme (one-sided
            # weighting), and it avoids the trap of baking a document-frequency
            # snapshot into vectors written early in a build.
            self._vecs[eu.unit_id] = self._embed(surface, use_idf=False, feats=feats)
            self._meta[eu.unit_id] = {
                "h": str(eu.indexing_hash),
                "d": eu.document_id,
                "s": [eu.unit.provenance.span.start, eu.unit.provenance.span.end],
                "u": eu.unit.provenance.source_uri,
                "t": surface,
                "f": {k: _jsonable(v) for k, v in eu.fields().items()},
            }
            written += 1
        if written:
            self._matrix = None
        self._dirty = True
        return IndexWriteReceipt(written=written, skipped=skipped)

    def delete(self, unit_ids: Sequence[UnitId], ctx: StageContext) -> int:
        n = 0
        for uid in unit_ids:
            if self._vecs.pop(uid, None) is not None:
                self._meta.pop(uid, None)
                n += 1
        if n:
            self._matrix = None
        self._dirty = True
        return n

    def delete_document(self, document_id: DocumentId, ctx: StageContext) -> int:
        ids = [u for u, m in self._meta.items() if m["d"] == document_id]
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
        if not self._vecs:
            return RankedList(hits=(), source=self.name, query_text=query.text)
        q = self._embed(query.text)
        allowed = set(query.unit_ids) if query.unit_ids is not None else None
        filtered = query.filters is not None or allowed is not None

        if HAVE_NUMPY and not filtered:
            scored = self._search_numpy(q)
        else:
            # The definition. Also the path taken whenever a filter or an id
            # restriction narrows the candidate set, where the vectorised scan
            # over everything would be the slower option anyway.
            scored = []
            for uid, vec in self._vecs.items():
                if allowed is not None and uid not in allowed:
                    continue
                if query.filters is not None and not _passes(query.filters, self._meta[uid]["f"]):
                    continue
                scored.append((uid, sum(a * b for a, b in zip(q, vec, strict=True))))

        top = sorted(scored, key=lambda kv: (-kv[1], kv[0]))[: query.top_k]
        return RankedList(
            hits=tuple(
                Hit(
                    unit_id=UnitId(uid),
                    document_id=DocumentId(self._meta[uid]["d"]),
                    rank=rank,
                    score=score,
                    index=self.name,
                    provenance=Provenance(
                        document_id=DocumentId(self._meta[uid]["d"]),
                        span=Span(*self._meta[uid]["s"]),
                        source_uri=self._meta[uid]["u"],
                    ),
                    matched_text=self._meta[uid]["t"],
                )
                for rank, (uid, score) in enumerate(top, start=1)
            ),
            source=self.name,
            query_text=query.text,
            fingerprint=self.fingerprint().key(),
            total_candidates=len(scored),
        )

    def _search_numpy(self, q: list[float]) -> list[tuple[str, float]]:
        if self._matrix is None:
            self._matrix_ids = list(self._vecs)
            self._matrix = _np.asarray([self._vecs[u] for u in self._matrix_ids], dtype=_np.float32)
        scores = self._matrix @ _np.asarray(q, dtype=_np.float32)
        return list(zip(self._matrix_ids, scores.tolist(), strict=True))

    def stats(self) -> IndexStatsView:
        return IndexStatsView(
            unit_count=len(self._vecs),
            detail={
                "dim": self.dim,
                "features": len(self._df),
                "exact_search": True,
                "accelerated": HAVE_NUMPY,
            },
        )

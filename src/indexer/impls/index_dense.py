"""Dense index: a vector store with a swappable embedder, exact cosine search.

This file used to be one class doing two jobs. The store half -- write, delete,
filter, exact cosine, persistence -- is the same arithmetic whatever produces
the vectors, and it lives here. The modelling half moved to ``impls/embed.py``,
which is the seam the old docstring promised: *"a real dense index is this file
with ``_embed`` replaced by a model call, and nothing else in the frame
changes."* It now is, and nothing else in the frame changed.

Three indexes are registered, differing only in their embedder:

``hash_embedding``       The hashing trick. Offline, no download, deterministic,
                         and **not** a semantic model -- it cannot match "how do
                         I stop a request hanging" to "timeout" unless the words
                         overlap. Unchanged and bit-exact, because the numbers in
                         ``docs/ABLATION.md`` were produced by it.
``svd_embedding``        Latent Semantic Analysis. A space *fitted* on the
                         corpus, still offline. The strongest dense index this
                         library can run with no model download.
``sentence_transformer`` A real neural bi-encoder. The production choice, and
                         the only one trained on the relation retrieval needs.

Search is exact -- brute-force cosine over every candidate -- at every corpus
size, which is the accuracy ceiling rather than an approximation of it. An ANN
index trades some of that ceiling for latency that only matters past roughly a
hundred thousand vectors; ``configs/full.yaml`` names Qdrant for when it does.
The seam is this file, and nothing above it changes.
"""

from __future__ import annotations

import json
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
from indexer.impls.embed import (
    MULTILINGUAL_PRESETS,
    Embedder,
    VoyageEmbedder,
    build_embedder,
)
from indexer.impls.index_lexical import _passes
from indexer.io import atomic_write, decode_value, encode_value
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["HashEmbeddingIndex", "SentenceTransformerIndex", "SvdIndex", "VectorIndex"]

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


class VectorIndex(StageImpl):
    """A vector store over ``EnrichedUnit.indexing_text()``, embedder-agnostic.

    Membership is ``_meta``, not ``_vecs``. The two are the same set for an
    embedder that can embed on arrival, but an embedder that must be fitted on
    the corpus first has units before it has vectors, and every count, delete
    and emptiness check has to mean "units I hold" rather than "vectors I have
    computed so far".
    """

    STAGE, IMPL, VERSION = "index", "vector", "2"
    kind = "dense"
    EMBEDDER = "hash"

    def __init__(
        self, params: dict[str, Any], name: str = "dense", embedder: Embedder | None = None
    ) -> None:
        super().__init__(params)
        self.name = name
        self.embedder = embedder or build_embedder(self.EMBEDDER, params)
        self.path = Path(params["path"]) if params.get("path") else None
        self._vecs: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._matrix: Any = None  # numpy cache, invalidated on write
        self._matrix_ids: list[str] = []
        # True when vectors are missing or stale relative to `_meta`. Only a
        # fitted embedder can set it: everything else embeds on arrival.
        self._fit_dirty = False
        self._dirty = False
        if self.path:
            self._load()

    @property
    def dim(self) -> int:
        return self.embedder.dim

    def _load(self) -> None:
        if not (self.path and self.path.exists()):
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        stored = raw.get("identity")
        if stored is not None and stored != self.store_identity():
            # Built by another embedder or with other parameters. The ledger
            # restages every document when this index's fingerprint changes;
            # loading the old vectors would make every rewrite a skip and leave
            # the store holding vectors from a model that is no longer asked --
            # of a different dimension, after a `dim` change, so that every
            # query against it raised.
            self._dirty = True
            return
        self._meta = raw["meta"]
        for m in self._meta.values():
            m["f"] = decode_value(m.get("f", {}))
        self._vecs = raw.get("vecs", {})
        # `df` is the pre-seam layout, where the hashing embedder's document
        # frequencies sat at the top level. Read it so stores built before the
        # embedder split -- including the ablation's build artifacts -- still load.
        state = raw.get("embedder")
        if state is None and "df" in raw:
            state = {"df": raw["df"]}
        if state:
            self.embedder.load_state(state)
        if self.embedder.needs_fit:
            # Fitted state is not persisted; it is rederived from the surfaces,
            # which are. See `SvdEmbedder.state`.
            self._fit_dirty = True

    def _save(self) -> None:
        if not self.path:
            return
        payload: dict[str, Any] = {
            "identity": self.store_identity(),
            "meta": {uid: {**m, "f": encode_value(m["f"])} for uid, m in self._meta.items()},
            "embedder": self.embedder.state(),
        }
        # Vectors derived from a fitted embedder are not persisted: they are a
        # function of the surfaces, which are in `meta`, and storing them would
        # double the file for state that is refitted on load anyway.
        payload["vecs"] = {} if self.embedder.needs_fit else self._vecs
        atomic_write(self.path, json.dumps(payload).encode())

    # ------------------------------------------------------------------ write

    def upsert(self, units: Sequence[EnrichedUnit], ctx: StageContext) -> IndexWriteReceipt:
        written = skipped = 0
        pending: list[tuple[str, str]] = []
        for eu in units:
            prev = self._meta.get(eu.unit_id)
            record = str(eu.record_hash)
            if prev is not None and prev.get("rh") == record:
                skipped += 1
                continue
            surface = eu.indexing_text()
            surface_hash = str(eu.indexing_hash)
            self._meta[eu.unit_id] = {
                "h": surface_hash,
                "rh": record,
                "d": eu.document_id,
                "s": [eu.unit.provenance.span.start, eu.unit.provenance.span.end],
                "u": eu.unit.provenance.source_uri,
                "t": surface,
                "f": dict(eu.filter_fields()),
            }
            # The record changed; the vector only if the surface did. A moved
            # span or a corrected field must be written, and must not cost an
            # embedding call.
            if prev is None or prev.get("h") != surface_hash or eu.unit_id not in self._vecs:
                pending.append((eu.unit_id, surface))
            written += 1

        if pending:
            if self.embedder.needs_fit:
                # The space this unit belongs in does not exist yet; the fit
                # happens once, at flush or at the next search.
                self._fit_dirty = True
            else:
                for uid, vec in zip(
                    (u for u, _ in pending),
                    self.embedder.embed_documents([s for _, s in pending]),
                    strict=True,
                ):
                    self._vecs[uid] = vec
            self._matrix = None
        self._dirty = True
        return IndexWriteReceipt(written=written, skipped=skipped)

    def delete(self, unit_ids: Sequence[UnitId], ctx: StageContext) -> int:
        n = 0
        for uid in unit_ids:
            if self._meta.pop(uid, None) is not None:
                self._vecs.pop(uid, None)
                n += 1
        if n:
            self._matrix = None
            if self.embedder.needs_fit:
                self._fit_dirty = True
        self._dirty = True
        return n

    def delete_document(self, document_id: DocumentId, ctx: StageContext) -> int:
        ids = [u for u, m in self._meta.items() if m["d"] == document_id]
        return self.delete([UnitId(i) for i in ids], ctx)

    def _ensure_vectors(self) -> None:
        """Materialise vectors for a fitted embedder.

        Lazy, and invalidated by writes, exactly like the numpy matrix below it.
        That is what keeps the ``Index`` contract honest: a query between
        batches fits on what has been written so far and returns correct
        results, rather than returning nothing until someone calls ``flush``.
        """
        if not self._fit_dirty:
            return
        ids = list(self._meta)
        surfaces = [self._meta[u]["t"] for u in ids]
        self.embedder.fit(surfaces)
        self._vecs = dict(zip(ids, self.embedder.embed_documents(surfaces), strict=True))
        self._fit_dirty = False
        self._matrix = None

    def flush(self) -> None:
        """Persist. Called once per build, not once per batch.

        Writing the whole file on every upsert is quadratic in corpus size --
        the reason this capability exists at all.
        """
        self._ensure_vectors()
        if self._dirty:
            self._save()
            self._dirty = False

    # ------------------------------------------------------------------- read

    def search(self, query: IndexQuery, ctx: StageContext) -> RankedList:
        if not self._meta:
            return RankedList(hits=(), source=self.name, query_text=query.text)
        self._ensure_vectors()
        q = self.embedder.embed_query(query.text, corpus_size=len(self._meta))
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
        if not self._matrix_ids:
            return []
        scores = self._matrix @ _np.asarray(q, dtype=_np.float32)
        return list(zip(self._matrix_ids, scores.tolist(), strict=True))

    def stats(self) -> IndexStatsView:
        return IndexStatsView(
            unit_count=len(self._meta),
            detail={
                **self.embedder.describe(),
                "exact_search": True,
                "accelerated": HAVE_NUMPY,
                "pending_fit": self._fit_dirty,
            },
        )


# ------------------------------------------------------------- registrations


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
    version="2",
    params_model=dataclass_params(HashEmbeddingParams),
    summary=(
        "Deterministic hashed embeddings, exact cosine. Offline and reproducible; "
        "NOT a semantic model -- absolute numbers are not meaningful, deltas are."
    ),
)
def _make_hash_embedding(params: dict[str, Any], **kw: Any) -> HashEmbeddingIndex:
    return HashEmbeddingIndex(params, name=kw.get("name", "dense"))


class HashEmbeddingIndex(VectorIndex):
    """The hashing trick. Bit-exact vectors since version 1.

    Version 2 changed what is stored beside the vectors (the record hash, typed
    filter fields, the store identity), not the arithmetic: the regression test
    pinning the vectors is unchanged.

    An honest framing of what this is: a hashed bag of n-grams projected into a
    fixed-dimensional space. It is a real vector index -- exact cosine, with all
    the write, delete and filter semantics a production store needs -- but it is
    **not** a semantic embedding model.

    Why it is still the reference path: it runs offline, in CI, with no model
    download and no credentials, which is what makes the pipeline testable and
    the ablation reproducible. The cost is that **absolute** dense-retrieval
    numbers from it mean nothing. For a dense index that is offline *and*
    learned, use ``svd_embedding``; for a real one, ``sentence_transformer``.
    """

    STAGE, IMPL, VERSION = "index", "hash_embedding", "2"
    EMBEDDER = "hash"


@dataclass(frozen=True, slots=True)
class SvdEmbeddingParams:
    dim: int = 512
    min_df: int = 2
    use_char_ngrams: bool = False
    char_ngram: int = 4
    random_state: int = 0
    path: str = ""


@register(
    "index",
    "svd_embedding",
    version="2",
    params_model=dataclass_params(SvdEmbeddingParams),
    summary=(
        "Latent Semantic Analysis: TF-IDF then truncated SVD. Offline, no download, "
        "and a learned space rather than a fixed projection."
    ),
    requires=("scipy",),
)
def _make_svd_embedding(params: dict[str, Any], **kw: Any) -> SvdIndex:
    return SvdIndex(params, name=kw.get("name", "dense"))


class SvdIndex(VectorIndex):
    """LSA. Offline, learned, and fitted once per build.

    The fit is deferred: units are held until a search or a flush needs vectors,
    then the whole corpus is factorised at once. That is not an optimisation, it
    is the only order that works -- there is no space to embed the first unit
    into until the last one has been seen.
    """

    STAGE, IMPL, VERSION = "index", "svd_embedding", "2"
    EMBEDDER = "svd"


@dataclass(frozen=True, slots=True)
class SentenceTransformerIndexParams:
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    device: str = "cpu"
    max_seq_length: int | None = None
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    document_prefix: str = ""
    truncate_dim: int | None = None
    dim: int = 0
    path: str = ""


@register(
    "index",
    "sentence_transformer",
    version="2",
    params_model=dataclass_params(SentenceTransformerIndexParams),
    summary=(
        "Neural bi-encoder (BGE/E5/MiniLM family). The production dense index; "
        "needs a model download."
    ),
    requires=("sentence-transformers",),
)
def _make_sentence_transformer(params: dict[str, Any], **kw: Any) -> SentenceTransformerIndex:
    return SentenceTransformerIndex(
        params, name=kw.get("name", "dense"), embedder=kw.get("embedder")
    )


class SentenceTransformerIndex(VectorIndex):
    """A real neural bi-encoder behind the same store.

    This is the drop-in the rest of the library was shaped around. Everything
    that makes it different from ``hash_embedding`` is inside ``_embed`` --
    which is now an object rather than a method, and is the only thing that
    changes.
    """

    STAGE, IMPL, VERSION = "index", "sentence_transformer", "2"
    EMBEDDER = "sentence_transformer"


# ------------------------------------------------------------- multilingual


@dataclass(frozen=True, slots=True)
class MultilingualIndexParams:
    #: bge-m3 | multilingual-e5-large | multilingual-e5-base | multilingual-e5-small.
    preset: str = "multilingual-e5-base"
    batch_size: int = 32
    device: str = "cpu"
    #: Overrides the preset's, when set.
    max_seq_length: int | None = None
    truncate_dim: int | None = None
    path: str = ""

    def __post_init__(self) -> None:
        if self.preset not in MULTILINGUAL_PRESETS:
            raise ValueError(
                f"preset must be one of {sorted(MULTILINGUAL_PRESETS)}, not {self.preset!r}"
            )


@register(
    "index",
    "multilingual_embedding",
    version="1",
    params_model=dataclass_params(MultilingualIndexParams),
    summary=(
        "Neural bi-encoder trained on Italian among ~100 languages (bge-m3, "
        "multilingual-e5), with each model's own query and passage prefixes."
    ),
    requires=("sentence-transformers",),
)
def _make_multilingual(params: dict[str, Any], **kw: Any) -> MultilingualIndex:
    return MultilingualIndex(params, name=kw.get("name", "dense"), embedder=kw.get("embedder"))


class MultilingualIndex(VectorIndex):
    """``sentence_transformer`` with the model chosen for Italian.

    A preset rather than a model name, because a multilingual model used without
    its prefixes -- E5's "query: " and "passage: " -- loses a measurable part of
    its quality, and nothing warns when they are missing.
    """

    STAGE, IMPL, VERSION = "index", "multilingual_embedding", "1"
    EMBEDDER = "sentence_transformer"

    def __init__(
        self, params: dict[str, Any], name: str = "dense", embedder: Embedder | None = None
    ) -> None:
        preset = MULTILINGUAL_PRESETS[str(params.get("preset", "multilingual-e5-base"))]
        resolved = {**preset, **{k: v for k, v in params.items() if v is not None}}
        resolved.pop("preset", None)
        super().__init__(
            params, name=name, embedder=embedder or build_embedder(self.EMBEDDER, resolved)
        )


# ------------------------------------------------------------------ voyage


@dataclass(frozen=True, slots=True)
class VoyageIndexParams:
    model: str = "voyage-3.5"
    #: Truncated output for the models that support it (256, 512, 1024, 2048).
    output_dimension: int | None = None
    batch_size: int = 64
    #: Where the key is read from. The key itself is never a parameter.
    api_key_env: str = "VOYAGE_API_KEY"
    base_url: str = "https://api.voyageai.com/v1/embeddings"
    max_retries: int = 4
    path: str = ""


@register(
    "index",
    "voyage_embedding",
    version="1",
    params_model=dataclass_params(VoyageIndexParams),
    summary=(
        "Voyage AI embeddings over HTTP: multilingual, nothing to host. Sends every "
        "chunk to a third party -- needs a GDPR basis."
    ),
)
def _make_voyage(params: dict[str, Any], **kw: Any) -> VoyageIndex:
    embedder = kw.get("embedder") or VoyageEmbedder(
        params, transport=kw.get("transport"), sleep=kw.get("sleep")
    )
    return VoyageIndex(params, name=kw.get("name", "dense"), embedder=embedder)


class VoyageIndex(VectorIndex):
    STAGE, IMPL, VERSION = "index", "voyage_embedding", "1"
    EMBEDDER = "voyage"

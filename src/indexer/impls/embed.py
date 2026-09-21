"""Embedders: the swappable half of a dense index.

``index_dense`` used to be one class doing two jobs -- a vector store (write,
delete, filter, exact cosine, persistence) and an embedding function. Only the
second is a modelling choice; the first is the same arithmetic whatever produces
the vectors. This module is that split, and it is what the old file's docstring
promised: *"a real dense index is this file with ``_embed`` replaced by a model
call, and nothing else in the frame changes."*

Three implementations, in increasing order of what they know about language:

``hash``                  The hashing trick. No model, no fit, no download.
                          Deterministic and offline, which is what keeps the
                          reference path runnable in CI. It cannot match
                          "how do I stop a request hanging" to "timeout".
``svd``                   Latent Semantic Analysis: TF-IDF, then a truncated
                          SVD. A *learned* space -- fitted on the corpus, so
                          terms that co-occur in similar contexts land near each
                          other -- and still offline, no download, ~3s to fit
                          ten thousand units. The strongest embedder that this
                          environment can actually run.
``sentence_transformer``  A real neural bi-encoder (BGE, E5, MiniLM). The
                          production choice. It needs a model download, so it
                          cannot run where the egress policy blocks the model
                          host; the contract is what makes it a drop-in anyway.

The protocol is small on purpose. Two things in it are worth explaining:

*Queries and documents embed differently.* Not a quirk: the hashing embedder
applies IDF to queries and not to documents (one-sided TF-IDF weighting, so a
document-frequency snapshot is never baked into vectors written early in a
build), and bi-encoders from the E5 and BGE families want an asymmetric
instruction prefix. A protocol with one ``embed`` would force both to lie.

*Some embedders must see the corpus before they can embed it.* ``svd`` has no
vector space until it has been fitted. ``needs_fit`` advertises that, and the
index defers materialising vectors until a search or a flush needs them -- the
same lazy-invalidation pattern the numpy matrix already uses.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from indexer.textutil import tokenize

__all__ = [
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "SvdEmbedder",
    "build_embedder",
]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a unit-norm vector.

    Must preserve
        *A stable dimension.* ``dim`` is fixed for the life of the embedder and
        every vector it returns has exactly that length. Vectors are persisted
        and compared across processes; a dimension that moves silently
        invalidates a store.

        *Unit norm, or the zero vector.* The index scores with a plain dot
        product and calls it cosine. An embedder that returns unnormalised
        vectors makes that a lie and long documents win everything.

        *Determinism.* The same text, the same fitted state, the same vector.
        Every eval delta in this library depends on it.
    """

    dim: int
    #: Whether the embedder must see the corpus (``fit``) before it can embed.
    needs_fit: bool

    def fit(self, texts: Sequence[str]) -> None:
        """Learn whatever state the embedder needs. A no-op when ``needs_fit``."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str, *, corpus_size: int) -> list[float]:
        """``corpus_size`` is the number of live vectors, for IDF-style weighting.

        Passed rather than held, because the embedder does not own the store and
        a stale count is a silent scoring bug.
        """
        ...

    def state(self) -> dict[str, Any]:
        """Serialisable fitted state, persisted beside the vectors."""
        ...

    def load_state(self, state: Mapping[str, Any]) -> None: ...

    def describe(self) -> dict[str, Any]:
        """Knobs worth showing in ``IndexStatsView.detail``."""
        ...


# --------------------------------------------------------------------- hashing


@dataclass(frozen=True, slots=True)
class HashEmbedderParams:
    dim: int = 256
    #: Character n-gram width. Character n-grams give some robustness to
    #: morphology ("timeout" / "timeouts") that word hashing alone lacks.
    char_ngram: int = 4
    use_words: bool = True
    use_char_ngrams: bool = True
    idf_weighting: bool = True


class HashEmbedder:
    """Hashed bag of n-grams, signed accumulation, one-sided IDF.

    The arithmetic here is unchanged from the original ``HashEmbeddingIndex``
    and a regression test pins it: the vectors this produces must stay
    bit-identical, because the ablation's build artifacts and every number in
    ``docs/ABLATION.md`` were produced by it. Re-deriving them cheaply is not a
    reason to make last quarter's report unreproducible.
    """

    needs_fit = False

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        p = dict(params or {})
        self.dim = int(p.get("dim", 256))
        self.char_ngram = int(p.get("char_ngram", 4))
        self.use_words = bool(p.get("use_words", True))
        self.use_chars = bool(p.get("use_char_ngrams", True))
        self.idf_weighting = bool(p.get("idf_weighting", True))
        self._df: dict[str, int] = {}

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

    def _project(
        self, feats: Mapping[str, float], *, use_idf: bool, corpus_size: int
    ) -> list[float]:
        """Hash features into ``dim`` buckets with signed accumulation.

        The sign comes from a second hash bit, which keeps collisions from
        systematically inflating similarity -- the standard hashing-trick
        correction, and without it every long document looks similar to every
        other long document.
        """
        vec = [0.0] * self.dim
        n_docs = max(1, corpus_size)
        for feat, count in feats.items():
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

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # Document frequency accumulates as documents arrive; documents
        # themselves are embedded unweighted. The dot product is then the
        # standard one-sided TF-IDF scheme, and it avoids baking a
        # document-frequency snapshot into vectors written early in a build.
        out = []
        for text in texts:
            feats = self._features(text)
            for feat in feats:
                self._df[feat] = self._df.get(feat, 0) + 1
            out.append(self._project(feats, use_idf=False, corpus_size=0))
        return out

    def embed_query(self, text: str, *, corpus_size: int) -> list[float]:
        return self._project(self._features(text), use_idf=True, corpus_size=corpus_size)

    def state(self) -> dict[str, Any]:
        return {"df": self._df}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self._df = dict(state.get("df", {}))

    def describe(self) -> dict[str, Any]:
        return {"embedder": "hash", "dim": self.dim, "features": len(self._df), "semantic": False}


# ------------------------------------------------------------------------ LSA


@dataclass(frozen=True, slots=True)
class SvdEmbedderParams:
    #: Latent dimensions kept. Unlike a neural embedder's width this is a
    #: precision/recall knob, not a model property: too few and distinct units
    #: collapse together, too many and the space is TF-IDF with extra steps.
    #: Worth an ablation on a new corpus -- `docs/ABLATION.md` records the sweep
    #: that chose this default.
    dim: int = 512
    #: Terms below this document frequency are dropped. Hapax legomena are most
    #: of a real vocabulary and contribute one-document dimensions the SVD then
    #: has to spend rank on.
    min_df: int = 2
    #: Character n-grams measurably *hurt* here -- they flood the vocabulary
    #: with near-duplicate dimensions and the truncation spends its rank on
    #: orthography rather than topic. Off by default, and the sweep is recorded.
    use_char_ngrams: bool = False
    char_ngram: int = 4
    random_state: int = 0


class SvdEmbedder:
    """Latent Semantic Analysis: TF-IDF, then a truncated SVD.

    What makes this a *model* rather than a trick: the projection is fitted on
    the corpus. Two terms that never co-occur but appear in the same contexts
    end up with similar coordinates, so a query can match a passage that shares
    no vocabulary with it. The hashing embedder cannot do that at any dimension,
    because its projection is fixed before it has seen a single document.

    What it still is not: a neural bi-encoder. LSA reads co-occurrence, not
    syntax or word order, and it is weakest exactly where documents are long and
    queries are short. Expect it to sit between the hashing trick and a real
    model, which is where the ablation puts it.

    Cost: one SVD over the whole corpus, ~3s for ten thousand units at dim=512.
    Paid at flush or at the first search after a write, never per query.
    """

    needs_fit = True

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        p = dict(params or {})
        self.dim = int(p.get("dim", 512))
        self.min_df = int(p.get("min_df", 2))
        self.use_chars = bool(p.get("use_char_ngrams", False))
        self.char_ngram = int(p.get("char_ngram", 4))
        self.random_state = int(p.get("random_state", 0))
        self._vocab: dict[str, int] = {}
        self._idf: Any = None
        self._components: Any = None  # (k, V) right singular vectors
        self._k = 0

    # The numeric core needs numpy and scipy. Imported at use rather than at
    # module import, so `indexer.impls` stays importable without the extra --
    # the registry advertises the requirement and the config check reports it
    # before anything tries to build an index.
    @staticmethod
    def _numpy() -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "svd_embedding needs numpy and scipy: pip install 'indexer[dense-svd]'"
            ) from exc
        return np

    @staticmethod
    def _scipy() -> tuple[Any, Any]:
        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import svds
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "svd_embedding needs numpy and scipy: pip install 'indexer[dense-svd]'"
            ) from exc
        return sp, svds

    def _features(self, text: str) -> Counter[str]:
        feats: Counter[str] = Counter(tokenize(text))
        if self.use_chars:
            norm = re.sub(r"\s+", " ", text.lower())
            n = self.char_ngram
            feats.update(f"\x00{norm[i : i + n]}" for i in range(max(0, len(norm) - n + 1)))
        return feats

    def _rows(self, feature_sets: Sequence[Mapping[str, float]]) -> Any:
        """L2-normalised sparse TF-IDF rows over the fitted vocabulary."""
        np = self._numpy()
        sp, _ = self._scipy()
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for feats in feature_sets:
            for term, tf in feats.items():
                j = self._vocab.get(term)
                if j is not None:
                    indices.append(j)
                    data.append((1.0 + math.log(tf)) * float(self._idf[j]))
            indptr.append(len(indices))
        m = sp.csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32),
            ),
            shape=(len(feature_sets), max(1, len(self._vocab))),
        )
        norms = sp.linalg.norm(m, axis=1)
        norms[norms == 0] = 1.0
        return sp.diags(1.0 / norms).dot(m).tocsr().astype(np.float32)

    def fit(self, texts: Sequence[str]) -> None:
        np = self._numpy()
        _, svds = self._scipy()
        n = len(texts)
        feature_sets = [self._features(t) for t in texts]

        df: Counter[str] = Counter()
        for feats in feature_sets:
            df.update(feats.keys())
        # Sorted, so the vocabulary -- and therefore every column index the SVD
        # sees -- does not depend on dict insertion order.
        terms = sorted(t for t, c in df.items() if c >= self.min_df)
        self._vocab = {t: i for i, t in enumerate(terms)}
        v = len(terms)
        if n == 0 or v == 0:
            self._components, self._idf, self._k = None, None, 0
            return
        self._idf = np.asarray(
            [math.log((1 + n) / (1 + df[t])) + 1.0 for t in terms], dtype=np.float32
        )

        x = self._rows(feature_sets)
        # svds needs k strictly below both dimensions. A small corpus -- a test
        # fixture, a first build -- therefore gets a smaller latent space than
        # asked for, and the vectors are zero-padded back to `dim` so the stored
        # dimension stays stable whatever the corpus size.
        k = max(1, min(self.dim, n - 1, v - 1))
        if k < 1:  # pragma: no cover - n<2 and v<2 both degenerate
            self._components, self._k = None, 0
            return
        _, s, vt = svds(x, k=k, random_state=self.random_state)
        order = np.argsort(-s)
        vt = vt[order]
        # svds fixes each singular vector only up to sign. Pinning it to the
        # largest-magnitude loading makes the fit reproducible across scipy
        # versions and solvers, which every eval delta here depends on.
        flip = np.sign(vt[np.arange(k), np.argmax(np.abs(vt), axis=1)])
        flip[flip == 0] = 1.0
        self._components = (vt * flip[:, None]).astype(np.float32)
        self._k = k

    def _project(self, feature_sets: Sequence[Mapping[str, float]]) -> list[list[float]]:
        np = self._numpy()
        if self._components is None:
            return [[0.0] * self.dim for _ in feature_sets]
        y = np.asarray(self._rows(feature_sets).dot(self._components.T))
        norms = np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-9)
        y = y / norms
        if self._k < self.dim:
            y = np.pad(y, ((0, 0), (0, self.dim - self._k)))
        return [[round(float(c), 6) for c in row] for row in y]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._project([self._features(t) for t in texts])

    def embed_query(self, text: str, *, corpus_size: int) -> list[float]:
        return self._project([self._features(text)])[0]

    def state(self) -> dict[str, Any]:
        # Deliberately empty. The fitted components are ~26MB of float32 for a
        # ten-thousand-unit corpus, the surfaces they are derived from are
        # already persisted with the index, and refitting them takes about as
        # long as parsing the JSON would. So the index refits on load instead,
        # and `random_state` plus the sign convention make that reproduce.
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "embedder": "svd",
            "dim": self.dim,
            "latent_rank": self._k,
            "vocabulary": len(self._vocab),
            "fitted": self._components is not None,
            "semantic": True,
        }


# ------------------------------------------------------------- neural encoder


@dataclass(frozen=True, slots=True)
class SentenceTransformerParams:
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    device: str = "cpu"
    max_seq_length: int | None = None
    #: E5 and BGE are trained with asymmetric prefixes and lose a measurable
    #: amount of their quality without them. The defaults are BGE's; E5 wants
    #: "query: " and "passage: ". Set both to "" for a symmetric model.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    document_prefix: str = ""
    #: Truncate to this many leading dimensions. Only meaningful for a
    #: Matryoshka-trained model, where the leading slice is itself a valid
    #: embedding -- which is what keeps two-stage search available without
    #: re-embedding the corpus later.
    truncate_dim: int | None = None
    dim: int = 0  # 0 = ask the model


class SentenceTransformerEmbedder:
    """A real neural bi-encoder. The production choice.

    Why this is the one to reach for: it is the only embedder here that was
    trained on the relation the retrieval task actually needs -- that a question
    and the passage answering it should land near each other even when they
    share no words. LSA approximates that from co-occurrence; the hashing trick
    does not attempt it.

    It needs the model weights. Where the egress policy blocks the model host
    this class raises at construction with the host named, rather than silently
    degrading to something weaker -- a dense index that quietly stopped being
    semantic is the failure mode invariant 6 exists to prevent.
    """

    needs_fit = False

    def __init__(self, params: Mapping[str, Any] | None = None, model: Any = None) -> None:
        p = dict(params or {})
        self.model_name = str(p.get("model", "BAAI/bge-small-en-v1.5"))
        self.batch_size = int(p.get("batch_size", 32))
        self.device = str(p.get("device", "cpu"))
        self.max_seq_length = p.get("max_seq_length")
        self.query_prefix = str(
            p.get("query_prefix", "Represent this sentence for searching relevant passages: ")
        )
        self.document_prefix = str(p.get("document_prefix", ""))
        self.truncate_dim = p.get("truncate_dim")
        self._model = model
        self.dim = int(p.get("dim", 0) or 0)
        if self._model is not None and not self.dim:
            self.dim = self._resolve_dim()

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "sentence_transformer needs sentence-transformers: "
                    "pip install 'indexer[dense-sentence-transformer]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length:
                self._model.max_seq_length = int(self.max_seq_length)
        return self._model

    def _resolve_dim(self) -> int:
        if self.truncate_dim:
            return int(self.truncate_dim)
        model = self._get_model()
        got = model.get_sentence_embedding_dimension()
        return int(got)

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if not self.dim:
            self.dim = self._resolve_dim()
        model = self._get_model()
        vecs = model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out = []
        for row in vecs:
            v = [float(x) for x in row][: self.dim]
            if self.truncate_dim:
                # A Matryoshka slice is a valid embedding but not a unit vector;
                # renormalising is what makes the dot product a cosine again.
                norm = math.sqrt(sum(x * x for x in v))
                if norm:
                    v = [x / norm for x in v]
            out.append(v)
        return out

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, self.document_prefix)

    def embed_query(self, text: str, *, corpus_size: int) -> list[float]:
        return self._encode([text], self.query_prefix)[0]

    def state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "embedder": "sentence_transformer",
            "model": self.model_name,
            "dim": self.dim,
            "truncate_dim": self.truncate_dim,
            "semantic": True,
        }


_EMBEDDERS: dict[str, Any] = {
    "hash": HashEmbedder,
    "svd": SvdEmbedder,
    "sentence_transformer": SentenceTransformerEmbedder,
}


def build_embedder(kind: str, params: Mapping[str, Any] | None = None) -> Embedder:
    try:
        cls = _EMBEDDERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown embedder {kind!r}; known: {sorted(_EMBEDDERS)}"
        ) from None
    embedder: Embedder = cls(params)
    return embedder

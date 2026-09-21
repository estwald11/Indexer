"""The embedder seam: the contract, and the promise that hashing did not move.

``index_dense`` was one class doing two jobs until the embedder was split out of
it. These tests exist to make that split provable rather than plausible: the
hashing arithmetic is pinned to literal vectors, the store is exercised against
an embedder that must be fitted before it can embed anything, and the neural
path is driven with a stub so its contract is tested where its weights cannot be
downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from indexer.core.accounting import InMemoryAccountant
from indexer.core.cache import NullCache
from indexer.core.ids import DocumentId, UnitId
from indexer.core.provenance import Provenance, Span
from indexer.core.registry import resolve
from indexer.core.stages import IndexQuery, StageContext
from indexer.core.unit import EnrichedUnit, Unit
from indexer.impls.embed import (
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
    SvdEmbedder,
    build_embedder,
)
from indexer.impls.index_dense import HashEmbeddingIndex, SvdIndex

scipy = pytest.importorskip("scipy", reason="svd_embedding needs the dense-svd extra")


def _ctx() -> StageContext:
    return StageContext(cache=NullCache(), accountant=InMemoryAccountant())


def _unit(uid: str, text: str, doc: str = "d1", start: int = 0) -> EnrichedUnit:
    return EnrichedUnit(
        unit=Unit(
            unit_id=UnitId(uid),
            document_id=DocumentId(doc),
            text=text,
            provenance=Provenance(
                document_id=DocumentId(doc),
                span=Span(start, start + len(text)),
                source_uri=f"mem://{doc}",
            ),
        ),
        enrichments={},
    )


CORPUS = [
    "the read timeout controls how long a request waits for the server",
    "connection pooling reuses sockets across requests to the same host",
    "retries repeat a failed request with exponential backoff between attempts",
    "the socket timeout is distinct from the total deadline for a request",
    "session objects keep cookies and connection pools between calls",
    "a proxy forwards the request to the upstream server on your behalf",
]


class TestHashEmbedderIsUnchanged:
    """The numbers in docs/ABLATION.md were produced by this arithmetic.

    Pinned to literals rather than to a stored artifact: a regression guard that
    needs a 53MB fixture is a guard nobody runs.
    """

    def test_document_vectors_are_pinned(self) -> None:
        e = HashEmbedder({"dim": 8, "char_ngram": 4})
        assert e.embed_documents(
            ["the read timeout in seconds", "connection pooling and retries"]
        ) == [
            [0.1525, -0.1525, 0.1525, -0.76249, 0.305, -0.1525, -0.4575, 0.1525],
            [-0.62554, -0.41703, 0.41703, 0.41703, 0.0, 0.0, -0.20851, 0.20851],
        ]

    def test_query_vector_is_pinned(self) -> None:
        e = HashEmbedder({"dim": 8, "char_ngram": 4})
        e.embed_documents(["the read timeout in seconds", "connection pooling and retries"])
        assert e.embed_query("read timeout", corpus_size=2) == [
            0.0,
            -0.37796,
            0.0,
            -0.75593,
            0.0,
            -0.37796,
            0.0,
            0.37796,
        ]

    def test_documents_are_unweighted_and_queries_are_not(self) -> None:
        """One-sided TF-IDF: a document-frequency snapshot must never be baked
        into vectors written early in a build."""
        e = HashEmbedder({"dim": 32})
        first = e.embed_documents(["alpha beta"])[0]
        e.embed_documents(["alpha gamma", "alpha delta"])
        assert e.embed_documents(["alpha beta"])[0] == first

    def test_legacy_store_layout_still_loads(self, tmp_path: Path) -> None:
        """Stores written before the embedder split keep `df` at the top level."""
        path = tmp_path / "dense.json"
        path.write_text(
            json.dumps(
                {
                    "vecs": {"u1": [1.0] + [0.0] * 31},
                    "meta": {
                        "u1": {
                            "h": "h1",
                            "d": "d1",
                            "s": [0, 5],
                            "u": "mem://d1",
                            "t": "alpha",
                            "f": {},
                        }
                    },
                    "df": {"w:alpha": 3},
                }
            )
        )
        idx = HashEmbeddingIndex({"path": str(path), "dim": 32})
        assert idx.stats().unit_count == 1
        assert idx.embedder.state()["df"] == {"w:alpha": 3}


class TestEmbedderContract:
    """What every embedder must promise, whatever it knows about language."""

    @pytest.mark.parametrize(
        ("kind", "params"),
        [("hash", {"dim": 64}), ("svd", {"dim": 4, "min_df": 1})],
    )
    def test_unit_norm_and_stable_dimension(self, kind: str, params: dict[str, Any]) -> None:
        e = build_embedder(kind, params)
        e.fit(CORPUS)
        vecs = [*e.embed_documents(CORPUS), e.embed_query("read timeout", corpus_size=len(CORPUS))]
        for v in vecs:
            assert len(v) == params["dim"]
            norm = sum(x * x for x in v) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-3) or norm == pytest.approx(0.0)

    @pytest.mark.parametrize(("kind", "params"), [("hash", {"dim": 64}), ("svd", {"dim": 4})])
    def test_deterministic(self, kind: str, params: dict[str, Any]) -> None:
        a, b = build_embedder(kind, params), build_embedder(kind, params)
        a.fit(CORPUS)
        b.fit(CORPUS)
        assert a.embed_query("read timeout", corpus_size=6) == pytest.approx(
            b.embed_query("read timeout", corpus_size=6)
        )

    def test_all_implementations_satisfy_the_protocol(self) -> None:
        for kind in ("hash", "svd", "sentence_transformer"):
            assert isinstance(build_embedder(kind, {}), Embedder)

    def test_unknown_embedder_names_what_is_known(self) -> None:
        with pytest.raises(ValueError, match="unknown embedder 'bge'"):
            build_embedder("bge", {})


class TestSvdEmbedder:
    def test_latent_rank_is_clamped_to_the_corpus(self) -> None:
        """A fixture corpus is smaller than any sane `dim`; the store's vector
        width must not move because of it."""
        e = SvdEmbedder({"dim": 256, "min_df": 1})
        e.fit(CORPUS[:3])
        assert e.describe()["latent_rank"] <= 2  # k < n
        assert all(len(v) == 256 for v in e.embed_documents(CORPUS[:3]))

    def test_empty_corpus_does_not_explode(self) -> None:
        e = SvdEmbedder({"dim": 16})
        e.fit([])
        assert e.embed_query("anything", corpus_size=0) == [0.0] * 16

    def test_fit_is_reproducible_across_instances(self) -> None:
        """The sign of a singular vector is arbitrary; the convention pins it.

        Without it a refit -- which is what loading an SVD store does -- would
        return a different space, and every stored comparison with it."""
        a, b = SvdEmbedder({"dim": 4, "min_df": 1}), SvdEmbedder({"dim": 4, "min_df": 1})
        a.fit(CORPUS)
        b.fit(CORPUS)
        for x, y in zip(a.embed_documents(CORPUS), b.embed_documents(CORPUS), strict=True):
            assert x == pytest.approx(y)

    def test_places_related_passages_nearer_than_unrelated_ones(self) -> None:
        """The property that distinguishes a fitted space from a fixed one."""
        e = SvdEmbedder({"dim": 3, "min_df": 1})
        e.fit(CORPUS)
        vecs = e.embed_documents(CORPUS)
        dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
        # 0 and 3 are both about timeouts; 1 is about connection pooling.
        assert dot(vecs[0], vecs[3]) > dot(vecs[0], vecs[1])


class TestSvdIndexDefersItsFit:
    """An embedder with no space yet still has to satisfy the Index contract."""

    def _write(self, idx: SvdIndex) -> None:
        idx.upsert(
            [_unit(f"u{i}", t, start=i * 100) for i, t in enumerate(CORPUS)],
            _ctx(),
        )

    def test_search_between_batches_is_correct_without_a_flush(self) -> None:
        """ "Results may be stale on disk but must already be correct in memory."""
        idx = SvdIndex({"dim": 4, "min_df": 1})
        self._write(idx)
        assert idx.stats().detail["pending_fit"] is True
        hits = idx.search(IndexQuery(text="read timeout", top_k=3), _ctx()).hits
        assert len(hits) == 3
        assert idx.stats().detail["pending_fit"] is False

    def test_unit_count_is_right_before_the_fit(self) -> None:
        idx = SvdIndex({"dim": 4, "min_df": 1})
        self._write(idx)
        assert idx.stats().unit_count == len(CORPUS)

    def test_delete_invalidates_the_fit(self) -> None:
        idx = SvdIndex({"dim": 4, "min_df": 1})
        self._write(idx)
        idx.search(IndexQuery(text="timeout", top_k=1), _ctx())
        assert idx.delete([UnitId("u0")], _ctx()) == 1
        assert idx.stats().detail["pending_fit"] is True
        assert idx.stats().unit_count == len(CORPUS) - 1
        assert "u0" not in idx.search(IndexQuery(text="read timeout", top_k=9), _ctx()).unit_ids()

    def test_round_trip_refits_and_reproduces(self, tmp_path: Path) -> None:
        """Fitted state is not persisted, so a reload must rederive the same space."""
        path = tmp_path / "svd.json"
        a = SvdIndex({"dim": 4, "min_df": 1, "path": str(path)})
        self._write(a)
        before = a.search(IndexQuery(text="read timeout", top_k=6), _ctx())
        a.flush()

        stored = json.loads(path.read_text())
        assert stored["vecs"] == {}, "derived vectors should not be persisted"

        b = SvdIndex({"dim": 4, "min_df": 1, "path": str(path)})
        after = b.search(IndexQuery(text="read timeout", top_k=6), _ctx())
        assert after.unit_ids() == before.unit_ids()
        for x, y in zip(before.hits, after.hits, strict=True):
            assert x.score == pytest.approx(y.score, abs=1e-5)


class _StubModel:
    """Stands in for a SentenceTransformer, and records what it was asked."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.seen: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts: list[str], **kw: Any) -> list[list[float]]:
        self.seen.extend(texts)
        out = []
        for t in texts:
            v = [float(sum(ord(c) for c in t) % (i + 7)) + 1.0 for i in range(self.dim)]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out


class TestSentenceTransformerEmbedder:
    """The production path, driven with a stub: the weights cannot be fetched
    here, but the contract around them can still be held to account."""

    def test_asymmetric_prefixes_are_applied(self) -> None:
        stub = _StubModel()
        e = SentenceTransformerEmbedder(
            {"query_prefix": "Q: ", "document_prefix": "D: "}, model=stub
        )
        e.embed_documents(["alpha"])
        e.embed_query("beta", corpus_size=1)
        assert stub.seen == ["D: alpha", "Q: beta"]

    def test_dimension_comes_from_the_model(self) -> None:
        e = SentenceTransformerEmbedder({}, model=_StubModel(dim=6))
        assert e.dim == 6
        assert len(e.embed_documents(["alpha"])[0]) == 6

    def test_matryoshka_slice_is_renormalised(self) -> None:
        """A truncated slice is a valid embedding but not a unit vector, and the
        store scores with a plain dot product."""
        e = SentenceTransformerEmbedder({"truncate_dim": 2}, model=_StubModel(dim=8))
        v = e.embed_documents(["alpha"])[0]
        assert len(v) == 2
        assert sum(x * x for x in v) ** 0.5 == pytest.approx(1.0)

    def test_missing_dependency_names_the_extra(self) -> None:
        e = SentenceTransformerEmbedder({"model": "BAAI/bge-small-en-v1.5"})
        with pytest.raises(RuntimeError, match="dense-sentence-transformer"):
            e.embed_documents(["alpha"])


class TestRegistrations:
    """Every dense index is reachable by name, and declares what it needs."""

    @pytest.mark.parametrize(
        ("name", "requires"),
        [
            ("hash_embedding", ()),
            ("svd_embedding", ("scipy",)),
            ("sentence_transformer", ("sentence-transformers",)),
        ],
    )
    def test_registered_with_its_requirements(self, name: str, requires: tuple[str, ...]) -> None:
        reg = resolve("index", name)
        assert reg.requires == requires
        assert reg.summary

    def test_all_three_share_the_store(self) -> None:
        """The point of the seam: one store, three embedders."""
        from indexer.impls.index_dense import VectorIndex

        for name in ("hash_embedding", "svd_embedding", "sentence_transformer"):
            reg = resolve("index", name)
            assert issubclass(reg.factory({}, name="dense").__class__, VectorIndex)

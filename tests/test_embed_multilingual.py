"""Dense retrieval that reads Italian: multilingual presets and Voyage embeddings."""

from __future__ import annotations

import json
from typing import Any

import pytest

from indexer.core.errors import ConfigError
from indexer.core.registry import resolve
from indexer.impls.embed import VoyageEmbedder


class TestMultilingualPresets:
    def test_a_preset_brings_its_model_and_prefixes(self) -> None:
        idx = resolve("index", "multilingual_embedding").build({"preset": "multilingual-e5-base"})
        e = idx.embedder
        assert e.model_name == "intfloat/multilingual-e5-base"
        assert (e.query_prefix, e.document_prefix) == ("query: ", "passage: ")

    def test_bge_m3_needs_no_prefix(self) -> None:
        e = resolve("index", "multilingual_embedding").build({"preset": "bge-m3"}).embedder
        assert (e.model_name, e.query_prefix, e.document_prefix) == ("BAAI/bge-m3", "", "")
        assert e.max_seq_length == 1024

    def test_an_explicit_setting_beats_the_preset(self) -> None:
        e = (
            resolve("index", "multilingual_embedding")
            .build({"preset": "bge-m3", "max_seq_length": 512})
            .embedder
        )
        assert e.max_seq_length == 512

    def test_an_unknown_preset_is_a_config_error(self) -> None:
        with pytest.raises(ConfigError, match="preset must be one of"):
            resolve("index", "multilingual_embedding").build({"preset": "e5-klingon"})


class _Voyage:
    """Voyage's embeddings endpoint, answered locally."""

    def __init__(self, fail_first: int = 0, status: int = 429) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.fail_first = fail_first
        self.status = status

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        payload = json.loads(body or b"{}")
        self.requests.append(payload)
        self.headers.append(headers)
        if self.fail_first:
            self.fail_first -= 1
            return self.status, b'{"detail": "slow down"}'
        data = [
            {
                "index": i,
                "embedding": [3.0, 4.0] if payload["input_type"] == "query" else [1.0, 0.0],
            }
            for i in range(len(payload["input"]))
        ]
        return 200, json.dumps(
            {"data": list(reversed(data)), "usage": {"total_tokens": 7}}
        ).encode()


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")


def test_documents_and_queries_are_sent_as_what_they_are() -> None:
    api = _Voyage()
    e = VoyageEmbedder(
        {"model": "voyage-3.5", "batch_size": 2, "output_dimension": 2}, transport=api
    )
    docs = e.embed_documents(["a", "b", "c"])
    query = e.embed_query("q", corpus_size=3)
    assert [r["input_type"] for r in api.requests] == ["document", "document", "query"]
    assert [len(r["input"]) for r in api.requests] == [2, 1, 1]
    assert api.headers[0]["Authorization"] == "Bearer pa-test"
    assert docs == [[1.0, 0.0]] * 3
    assert query == pytest.approx([0.6, 0.8])  # normalised
    assert e.describe()["tokens_billed"] == 21


def test_rate_limits_are_retried_and_other_errors_raise() -> None:
    naps: list[float] = []
    ok = VoyageEmbedder({"output_dimension": 2}, transport=_Voyage(fail_first=2), sleep=naps.append)
    assert ok.embed_query("q", corpus_size=1) == pytest.approx([0.6, 0.8])
    assert naps == [1.0, 2.0]

    bad = VoyageEmbedder({"output_dimension": 2}, transport=_Voyage(fail_first=1, status=400))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        bad.embed_query("q", corpus_size=1)


def test_a_vector_of_the_wrong_size_is_refused() -> None:
    e = VoyageEmbedder({"model": "voyage-3.5"}, transport=_Voyage())  # expects 1024
    with pytest.raises(RuntimeError, match="the store holds 1024"):
        e.embed_documents(["a"])


def test_no_key_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY")
    api = _Voyage()
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageEmbedder({}, transport=api).embed_query("q", corpus_size=1)
    assert not api.requests


def test_the_key_is_not_part_of_what_the_index_is() -> None:
    reg = resolve("index", "voyage_embedding")
    assert reg.fingerprint({}) == reg.fingerprint({"api_key_env": "VOYAGE_API_KEY"})

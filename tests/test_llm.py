"""The shared model-calling conventions: price, cache minimums, request shape,
refusals, concurrency and batches. No network: a fake client answers."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from indexer.llm import (
    FALLBACK_BETA,
    THINKING_HEADROOM,
    Claude,
    LLMError,
    Usage,
    cache_min_tokens,
    cost_usd,
    request,
)
from llm_fakes import FakeClient, message


class TestPrices:
    def test_a_dated_model_id_is_priced_as_its_family(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost_usd("claude-haiku-4-5-20251001", usage) == pytest.approx(6.0)
        assert cost_usd("claude-opus-5", usage) == pytest.approx(30.0)
        # Opus 5.5 is its own entry, not Opus 5 by prefix.
        assert cost_usd("claude-opus-5-5", usage) == pytest.approx(24.0)

    def test_cache_reads_and_writes_are_priced_apart(self) -> None:
        usage = Usage(cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
        assert cost_usd("claude-haiku-4-5", usage) == pytest.approx(0.1 + 1.25)

    def test_a_batch_costs_half(self) -> None:
        usage = Usage(input_tokens=1_000_000)
        assert cost_usd("claude-sonnet-5", usage, batch=True) == pytest.approx(1.0)

    def test_configured_prices_win_and_unknown_models_cost_zero(self) -> None:
        usage = Usage(input_tokens=1_000_000)
        prices = {"claude-haiku-4-5": {"input": 0.8, "output": 4.0}}
        assert cost_usd("claude-haiku-4-5", usage, prices) == pytest.approx(0.8)
        assert cost_usd("some-other-model", usage) == 0.0


def test_cache_minimums_are_per_model() -> None:
    assert cache_min_tokens("claude-haiku-4-5") == 4096
    assert cache_min_tokens("claude-haiku-4-5-20251001") == 4096
    assert cache_min_tokens("claude-opus-5") == 512
    assert cache_min_tokens("claude-opus-5-5") == 512
    assert cache_min_tokens("claude-sonnet-5") == 1024
    assert cache_min_tokens("unheard-of") == 4096


class TestRequest:
    def test_effort_is_left_out_where_the_model_rejects_it(self) -> None:
        haiku = request(model="claude-haiku-4-5", max_tokens=100, content="x", effort="low")
        assert "output_config" not in haiku
        opus = request(model="claude-opus-5", max_tokens=100, content="x", effort="low")
        assert opus["output_config"] == {"effort": "low"}

    def test_a_model_that_thinks_by_default_gets_room_to(self) -> None:
        assert request(model="claude-haiku-4-5", max_tokens=100, content="x")["max_tokens"] == 100
        opus = request(model="claude-opus-5", max_tokens=100, content="x")
        assert opus["max_tokens"] == 100 + THINKING_HEADROOM

    def test_a_schema_becomes_a_json_output_format(self) -> None:
        schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        params = request(model="claude-haiku-4-5", max_tokens=10, content="x", schema=schema)
        assert params["output_config"]["format"] == {"type": "json_schema", "schema": schema}


class TestCall:
    def test_refusal_truncation_and_bad_json_are_errors_not_answers(self) -> None:
        schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        for answer, kind in (
            (message("", stop="refusal", category="bio"), "refusal"),
            (message("half an ans", stop="max_tokens"), "truncated"),
            (message("not json"), "invalid_output"),
        ):
            claude = Claude(FakeClient(lambda p, a=answer: a))
            with pytest.raises(LLMError) as err:
                claude.call(
                    request(model="claude-haiku-4-5", max_tokens=10, content="x", schema=schema)
                )
            assert err.value.kind == kind
            assert not err.value.fatal

    def test_refusals_fall_back_server_side_where_supported(self) -> None:
        client = FakeClient(lambda p: message("ok", model=p["model"]))
        Claude(client).call(request(model="claude-opus-5", max_tokens=10, content="x"))
        assert client.beta_calls[-1]["betas"] == [FALLBACK_BETA]
        assert client.beta_calls[-1]["fallbacks"] == "default"

        Claude(client).call(request(model="claude-haiku-4-5", max_tokens=10, content="x"))
        Claude(client, fallbacks="none").call(
            request(model="claude-opus-5", max_tokens=10, content="x")
        )
        assert len(client.beta_calls) == 1
        assert "fallbacks" not in client.calls[-1]

    def test_the_served_model_is_the_one_priced(self) -> None:
        client = FakeClient(
            lambda p: message("ok", model="claude-opus-4-8", input_tokens=1_000_000)
        )
        result = Claude(client).call(request(model="claude-opus-5", max_tokens=10, content="x"))
        assert result.model == "claude-opus-4-8"
        assert result.cost_usd == pytest.approx(5.0 + 50 * 25.0 / 1_000_000)

    def test_an_authentication_failure_is_fatal_and_a_server_error_is_not(self) -> None:
        class ApiError(Exception):
            def __init__(self, status: int) -> None:
                super().__init__(f"status {status}")
                self.status_code = status

        for status, fatal in ((401, True), (404, True), (500, False), (400, False)):
            claude = Claude(FakeClient(lambda p, s=status: ApiError(s)))
            with pytest.raises(LLMError) as err:
                claude.call(request(model="claude-haiku-4-5", max_tokens=10, content="x"))
            assert err.value.fatal is fatal


def test_the_first_request_warms_the_cache_before_the_rest_go() -> None:
    events: list[tuple[str, int]] = []
    lock = threading.Lock()
    active = [0, 0]  # now, max

    def respond(params: dict[str, Any]) -> Any:
        i = int(params["messages"][0]["content"])
        with lock:
            events.append(("start", i))
            active[0] += 1
            active[1] = max(active)
        time.sleep(0.02)
        with lock:
            events.append(("end", i))
            active[0] -= 1
        return message(f"answer {i}")

    claude = Claude(FakeClient(respond))
    requests = [request(model="claude-haiku-4-5", max_tokens=10, content=str(i)) for i in range(6)]
    results = claude.call_many(requests, max_concurrency=4, warm_first=True)

    assert events[:2] == [("start", 0), ("end", 0)]
    assert [r.text for r in results] == [f"answer {i}" for i in range(6)]  # type: ignore[union-attr]
    assert active[1] > 1  # the rest did go in parallel


def test_a_failed_request_does_not_sink_the_others() -> None:
    def respond(params: dict[str, Any]) -> Any:
        if params["messages"][0]["content"] == "2":
            return message("", stop="refusal")
        return message("fine")

    out = Claude(FakeClient(respond)).call_many(
        [request(model="claude-haiku-4-5", max_tokens=10, content=str(i)) for i in range(4)],
        max_concurrency=2,
    )
    assert [isinstance(r, LLMError) for r in out] == [False, False, True, False]


def test_batches_are_split_polled_and_read_back_by_custom_id() -> None:
    def respond(params: dict[str, Any]) -> Any:
        content = params["messages"][0]["content"]
        if content == "bad":
            return RuntimeError("invalid")
        return message(f"answer to {content}", input_tokens=1_000_000, output_tokens=0)

    client = FakeClient(respond)
    naps: list[float] = []
    requests = [
        (f"r{i}", request(model="claude-haiku-4-5", max_tokens=10, content=c))
        for i, c in enumerate(["a", "b", "bad", "c", "d"])
    ]
    out = Claude(client).batch(requests, poll_seconds=7, sleep=naps.append, max_requests=2)

    assert len(client.messages.batches.created) == 3  # type: ignore[attr-defined]
    assert naps == [7, 7, 7]  # each batch polled once before it ended
    assert out["r0"].text == "answer to a"  # type: ignore[union-attr]
    assert out["r0"].cost_usd == pytest.approx(0.5)  # type: ignore[union-attr]
    assert isinstance(out["r2"], LLMError) and out["r2"].kind == "batch_errored"
    assert not client.calls  # nothing went through the live endpoint


def test_an_sdk_too_old_for_a_parameter_stops_everything() -> None:
    """Every call would fail the same way; recording it once per unit would
    finish a build with nothing enriched."""
    client = FakeClient(
        lambda p: TypeError("create() got an unexpected keyword argument 'fallbacks'")
    )
    with pytest.raises(LLMError, match="set fallbacks: none") as err:
        Claude(client).call(request(model="claude-opus-5", max_tokens=10, content="x"))
    assert err.value.fatal

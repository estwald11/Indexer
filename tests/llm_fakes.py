"""A fake Anthropic client: the SDK's ``messages`` surface, answered by a function.

Enough of the surface for the model-backed stages -- ``messages.create``,
``beta.messages.create``, ``messages.batches`` -- with every request recorded,
so a test can assert what was sent as well as what was made of the answer.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

Responder = Callable[[dict[str, Any]], Any]


def message(
    text: str | None = None,
    *,
    data: Any = None,
    stop: str = "end_turn",
    model: str = "claude-haiku-4-5",
    input_tokens: int = 1000,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_write: int = 0,
    category: str | None = None,
) -> SimpleNamespace:
    """A response as the SDK returns one."""
    body = json.dumps(data) if data is not None else (text or "")
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=body)] if body else [],
        stop_reason=stop,
        stop_details=SimpleNamespace(category=category) if stop == "refusal" else None,
        model=model,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def prompt_of(params: dict[str, Any]) -> str:
    """Every text the request carries, system included, as one string."""
    parts: list[str] = []
    system = params.get("system")
    if isinstance(system, str):
        parts.append(system)
    for m in params.get("messages", []):
        content = m["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(b.get("text", "") for b in content)
    return "\n".join(parts)


def schema_of(params: dict[str, Any]) -> dict[str, Any]:
    fmt = (params.get("output_config") or {}).get("format") or {}
    schema: dict[str, Any] = fmt.get("schema") or {}
    return schema


class _Messages:
    def __init__(self, client: FakeClient, beta: bool) -> None:
        self._client = client
        self._beta = beta

    def create(self, **params: Any) -> Any:
        with self._client.lock:
            self._client.calls.append(params)
            if self._beta:
                self._client.beta_calls.append(params)
        answer = self._client.respond(params)
        # A responder returns an exception to have the call raise it, so one
        # responder serves both the live path and the batch path.
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Batches:
    def __init__(self, client: FakeClient) -> None:
        self._client = client
        self.created: dict[str, list[dict[str, Any]]] = {}
        self.polls: dict[str, int] = {}

    def create(self, *, requests: list[dict[str, Any]]) -> Any:
        bid = f"msgbatch_{len(self.created)}"
        self.created[bid] = list(requests)
        self.polls[bid] = 0
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid: str) -> Any:
        self.polls[bid] += 1
        status = "ended" if self.polls[bid] > 1 else "in_progress"
        return SimpleNamespace(id=bid, processing_status=status)

    def results(self, bid: str) -> Iterator[Any]:
        # Out of order on purpose: results are keyed by custom id, not position.
        for r in reversed(self.created[bid]):
            self._client.batched.append(r["params"])
            answer = self._client.respond(r["params"])
            if isinstance(answer, Exception):
                yield SimpleNamespace(
                    custom_id=r["custom_id"],
                    result=SimpleNamespace(
                        type="errored", error=SimpleNamespace(type="invalid_request")
                    ),
                )
                continue
            yield SimpleNamespace(
                custom_id=r["custom_id"],
                result=SimpleNamespace(type="succeeded", message=answer),
            )


class FakeClient:
    def __init__(self, respond: Responder) -> None:
        self.respond = respond
        self.calls: list[dict[str, Any]] = []
        self.beta_calls: list[dict[str, Any]] = []
        self.batched: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.messages = _Messages(self, beta=False)
        self.messages.batches = _Batches(self)  # type: ignore[attr-defined]
        self.beta = SimpleNamespace(messages=_Messages(self, beta=True))

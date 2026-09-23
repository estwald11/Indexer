"""Calling Claude: one convention for every model-backed stage.

Four stages call a model -- the contextualiser, the classifier, the field
extractor and the LLM router. They differ in their prompts and in nothing else
worth getting wrong four times. This module is the rest:

*Cost is measured.* Every call's usage, cache writes and reads included, is
priced into ``cost_usd`` from ``accounting.prices`` (list prices when a model
is not configured), so a build's manifest says what it spent instead of zero.

*A refusal or a truncated answer is an error.* Both arrive as HTTP 200. Reading
``content[0]`` of either indexed an empty summary, or half a JSON object, as if
it were the answer -- and the cache then served it forever. Both raise
``LLMError``, and the stage's error policy decides what happens.

*Output is structured.* A stage that wants data asks for it with a JSON schema
(``output_config.format``): the answer parses or the call fails. No regex over
prose, and no forced tool call, which some models reject.

*Refusals fall back server-side* on the models that support it (``fallbacks:
"default"``), so a false positive on one chunk of a sick-leave form does not
leave that chunk unindexed.

*Concurrency does not defeat the cache.* Requests that share a cached prefix
send one alone first, so it writes the cache the others then read. Firing them
all at once has every one of them pay the cache write.

*Batches cost half.* ``Claude.batch`` sends the same requests through the
Message Batches API -- for a first build of an archive, where nobody waits.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar

from indexer.core.accounting import StageFingerprint
from indexer.core.errors import EnrichmentIncomplete, IndexerError
from indexer.core.ids import hash_obj
from indexer.core.stages import EnrichContext
from indexer.core.unit import Enrichment, Unit
from indexer.plugin import StageImpl

__all__ = [
    "FALLBACK_BETA",
    "LIST_PRICES",
    "Claude",
    "LLMError",
    "LLMResult",
    "ModelEnricher",
    "Usage",
    "cache_min_tokens",
    "cost_usd",
    "estimate_tokens",
    "json_object",
    "nullable",
    "request",
    "supports_effort",
    "supports_server_fallbacks",
    "thinks_by_default",
]

#: Beta header for ``fallbacks: "default"`` (the array form uses another one).
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: USD per million tokens, first-party Claude API list prices (September 2026).
#: Only a fallback: ``accounting.prices`` in config wins, because prices change
#: and a stale constant misreports cost without anything looking wrong. Cache
#: reads default to 0.1x input and 5-minute cache writes to 1.25x.
LIST_PRICES: dict[str, dict[str, float]] = {
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.25},
    "claude-fable-5": {"input": 10.0, "output": 50.0},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}

#: The shortest prompt prefix a model will cache. A shorter prefix is not an
#: error: the marker is accepted and nothing is cached, so every call pays the
#: full document again. Not monotonic across generations -- Haiku 4.5 needs
#: eight times what Opus 5 does.
_CACHE_MIN_TOKENS: dict[str, int] = {
    "claude-opus-5": 512,  # and claude-opus-5-5
    "claude-fable-5": 512,  # and claude-fable-5-1
    "claude-mythos-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-opus-4-7": 2048,
    "claude-opus-4-6": 4096,
    "claude-opus-4-5": 4096,
    "claude-haiku-4-5": 4096,
}

#: Models that think (adaptively) when a request says nothing about thinking.
_THINKS_BY_DEFAULT = ("claude-opus-5", "claude-fable-5", "claude-mythos-5", "claude-sonnet-5")
#: Models that reject ``output_config.effort``.
_NO_EFFORT = ("claude-haiku-4-5", "claude-sonnet-4-5")
#: Models with server-side refusal fallbacks on the Claude API.
_SERVER_FALLBACKS = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")

#: Output budget added for a model that thinks by default. Thinking counts
#: against ``max_tokens``; without room for it a 160-token summary budget is
#: spent before the summary starts, and the call ends truncated. A ceiling, not
#: a cost: only generated tokens are billed.
THINKING_HEADROOM = 4096

#: Characters per token, for decisions that need an estimate before a call.
#: Deliberately high: it errs toward *under*-counting a document, which errs
#: toward one grouped call -- the cheaper mistake.
_CHARS_PER_TOKEN = 4.0


def _family(model: str, names: Iterable[str]) -> str | None:
    """The longest known model id that ``model`` is, or is a dated/versioned
    form of. ``claude-haiku-4-5-20251001`` is ``claude-haiku-4-5``."""
    model = model.removeprefix("anthropic.")
    best: str | None = None
    for name in names:
        if (model == name or model.startswith((name + "-", name + "@"))) and (
            best is None or len(name) > len(best)
        ):
            best = name
    return best


def cache_min_tokens(model: str) -> int:
    """Minimum cacheable prefix, in tokens. Unknown models get the largest."""
    name = _family(model, _CACHE_MIN_TOKENS)
    return _CACHE_MIN_TOKENS[name] if name else 4096


def thinks_by_default(model: str) -> bool:
    return _family(model, _THINKS_BY_DEFAULT) is not None


def supports_effort(model: str) -> bool:
    return _family(model, _NO_EFFORT) is None


def supports_server_fallbacks(model: str) -> bool:
    return _family(model, _SERVER_FALLBACKS) is not None


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- #
# usage, cost, results                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens one call consumed. ``input_tokens`` excludes cached tokens, as
    the API reports it; ``tokens_in`` is everything the model read."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def of(cls, message: Any) -> Usage:
        u = getattr(message, "usage", None)
        if u is None:
            return cls()
        return cls(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        )

    @property
    def tokens_in(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


def cost_usd(
    model: str,
    usage: Usage,
    prices: Mapping[str, Mapping[str, float]] | None = None,
    *,
    batch: bool = False,
) -> float:
    """What ``usage`` cost on ``model``. Zero for a model with no known price --
    reported as zero rather than guessed."""
    table: dict[str, Mapping[str, float]] = {**LIST_PRICES, **(prices or {})}
    name = _family(model, table)
    if name is None:
        return 0.0
    p = table[name]

    def price(key: str, default: float) -> float:
        # "input" or "input_per_mtok": configs have been written both ways.
        for k in (key, f"{key}_per_mtok"):
            if k in p:
                return float(p[k])
        return default

    inp = price("input", 0.0)
    total = (
        usage.input_tokens * inp
        + usage.output_tokens * price("output", 0.0)
        + usage.cache_read_tokens * price("cache_read", inp * 0.1)
        + usage.cache_write_tokens * price("cache_write", inp * 1.25)
    ) / 1_000_000
    return total * 0.5 if batch else total


@dataclass(frozen=True, slots=True)
class LLMResult:
    """A usable answer. ``data`` is the parsed JSON when a schema was asked for."""

    text: str
    data: Any
    usage: Usage
    model: str
    cost_usd: float


class LLMError(IndexerError):
    """A call that produced no usable answer.

    ``fatal`` marks failures that will fail every call the same way -- a bad
    key, a model id that does not exist, a missing package. A build stops on
    those instead of recording one "failed enrichment" per unit and finishing
    with nothing enriched.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "api",
        fatal: bool = False,
        usage: Usage | None = None,
        cost_usd: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.fatal = fatal
        self.usage = usage or Usage()
        self.cost_usd = cost_usd


# --------------------------------------------------------------------------- #
# requests                                                                     #
# --------------------------------------------------------------------------- #


def json_object(properties: Mapping[str, Any]) -> dict[str, Any]:
    """A closed JSON-schema object with every property required -- the shape
    structured outputs accept. Absence is expressed with ``nullable``."""
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [dict(schema), {"type": "null"}]}


def request(
    *,
    model: str,
    max_tokens: int,
    content: str | Sequence[Mapping[str, Any]],
    system: str | Sequence[Mapping[str, Any]] | None = None,
    schema: Mapping[str, Any] | None = None,
    effort: str = "",
) -> dict[str, Any]:
    """The parameters of one Messages API call -- the shape ``messages.create``
    takes and a batch request carries, so both paths send the same thing."""
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens) + (THINKING_HEADROOM if thinks_by_default(model) else 0),
        "messages": [
            {
                "role": "user",
                "content": content if isinstance(content, str) else [dict(b) for b in content],
            }
        ],
    }
    if system:
        params["system"] = system if isinstance(system, str) else [dict(b) for b in system]
    output: dict[str, Any] = {}
    if schema is not None:
        output["format"] = {"type": "json_schema", "schema": dict(schema)}
    if effort and supports_effort(model):
        output["effort"] = effort
    if output:
        params["output_config"] = output
    return params


# --------------------------------------------------------------------------- #
# the client                                                                   #
# --------------------------------------------------------------------------- #


class Claude:
    """An Anthropic client with the frame's conventions attached.

    ``client`` is anything with the SDK's ``messages`` surface -- tests pass a
    fake. By default an ``anthropic.Anthropic()`` is built on first use, so a
    stage can be constructed, fingerprinted and validated with neither the
    package installed nor a key set.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        prices: Mapping[str, Mapping[str, float]] | None = None,
        fallbacks: str = "default",
    ) -> None:
        self._client = client
        self.prices = {k: dict(v) for k, v in (prices or {}).items()}
        #: ``default`` or ``none``. Applied only where the model supports it,
        #: and never in a batch, which rejects the parameter.
        self.fallbacks = fallbacks

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise LLMError(
                    "model-backed stages need the `anthropic` package: pip install 'indexer[llm]'",
                    kind="setup",
                    fatal=True,
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def call(self, params: Mapping[str, Any]) -> LLMResult:
        model = str(params["model"])
        try:
            if self.fallbacks != "none" and supports_server_fallbacks(model):
                message = self.client.beta.messages.create(
                    **params, betas=[FALLBACK_BETA], fallbacks=self.fallbacks
                )
            else:
                message = self.client.messages.create(**params)
        except LLMError:
            raise
        except Exception as exc:
            # The SDK has already retried what is retryable (429, 5xx, network).
            # Authentication, permission and not-found fail every call alike.
            status = getattr(exc, "status_code", None)
            raise LLMError(
                f"{type(exc).__name__}: {exc}", kind="api", fatal=status in (401, 403, 404)
            ) from exc
        return self.result(message, params)

    def result(self, message: Any, params: Mapping[str, Any], *, batch: bool = False) -> LLMResult:
        """Check a response and read it. Stop reason first: a refusal has no
        answer and a truncated one has half of one."""
        usage = Usage.of(message)
        served = str(getattr(message, "model", "") or params["model"])
        cost = cost_usd(served, usage, self.prices, batch=batch)
        stop = getattr(message, "stop_reason", None)
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            raise LLMError(
                f"{served} declined ({category or 'no category'})",
                kind="refusal",
                usage=usage,
                cost_usd=cost,
            )
        if stop == "max_tokens":
            raise LLMError(
                f"answer truncated at max_tokens={params.get('max_tokens')}",
                kind="truncated",
                usage=usage,
                cost_usd=cost,
            )
        text = "".join(
            str(getattr(b, "text", ""))
            for b in getattr(message, "content", None) or ()
            if getattr(b, "type", None) == "text"
        ).strip()
        data: Any = None
        if "format" in (params.get("output_config") or {}):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"answer is not the JSON the schema asked for: {text[:80]!r}",
                    kind="invalid_output",
                    usage=usage,
                    cost_usd=cost,
                ) from exc
        return LLMResult(text=text, data=data, usage=usage, model=served, cost_usd=cost)

    def call_many(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        max_concurrency: int = 1,
        warm_first: bool = False,
    ) -> list[LLMResult | LLMError]:
        """Answer every request; a failed one yields its ``LLMError`` in place.
        Fatal errors raise.

        ``warm_first``: the requests share a cached prefix, so one goes alone
        and the rest follow in parallel -- reading the cache the first wrote.
        """
        if not requests:
            return []
        _ = self.client  # built once, before any worker thread needs it
        out: list[LLMResult | LLMError | None] = [None] * len(requests)

        def one(i: int) -> None:
            try:
                out[i] = self.call(requests[i])
            except LLMError as exc:
                if exc.fatal:
                    raise
                out[i] = exc

        start = 0
        if warm_first and max_concurrency > 1 and len(requests) > 1:
            one(0)
            start = 1
        rest = range(start, len(requests))
        if max_concurrency <= 1 or len(rest) <= 1:
            for i in rest:
                one(i)
        else:
            with ThreadPoolExecutor(max_workers=min(max_concurrency, len(rest))) as pool:
                for f in [pool.submit(one, i) for i in rest]:
                    f.result()
        return [r for r in out if r is not None]

    def batch(
        self,
        requests: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        poll_seconds: float = 60.0,
        timeout_seconds: float = 25 * 3600.0,
        sleep: Callable[[float], None] = time.sleep,
        progress: Callable[[str], None] | None = None,
        max_requests: int = 10_000,
        max_bytes: int = 200_000_000,
    ) -> dict[str, LLMResult | LLMError]:
        """Answer ``(custom_id, params)`` requests through the Message Batches
        API, at half price, keyed by custom id -- results come back in any order.

        Split to stay under the API's per-batch limits (100,000 requests,
        256 MB). A request that errored or expired yields an ``LLMError``; the
        caller decides whether it is retried.
        """
        by_id = {cid: dict(p) for cid, p in requests}
        if len(by_id) != len(requests):
            raise ValueError("batch custom ids must be unique")
        out: dict[str, LLMResult | LLMError] = {}
        for chunk in _chunked(list(by_id.items()), max_requests, max_bytes):
            created = self.client.messages.batches.create(
                requests=[{"custom_id": cid, "params": p} for cid, p in chunk]
            )
            waited = 0.0
            while True:
                status = self.client.messages.batches.retrieve(created.id)
                if status.processing_status == "ended":
                    break
                if waited >= timeout_seconds:
                    raise LLMError(
                        f"batch {created.id} still {status.processing_status} after {waited:.0f}s",
                        kind="batch_timeout",
                    )
                if progress:
                    progress(f"batch {created.id}: {status.processing_status}")
                sleep(poll_seconds)
                waited += poll_seconds
            for item in self.client.messages.batches.results(created.id):
                cid = str(item.custom_id)
                outcome = item.result
                if outcome.type == "succeeded":
                    try:
                        out[cid] = self.result(outcome.message, by_id[cid], batch=True)
                    except LLMError as exc:
                        out[cid] = exc
                else:
                    out[cid] = LLMError(
                        f"batch request {outcome.type}: {_batch_error(outcome)}",
                        kind=f"batch_{outcome.type}",
                    )
        return out


def _batch_error(outcome: Any) -> str:
    err = getattr(outcome, "error", None)
    inner = getattr(err, "error", err)
    return str(getattr(inner, "type", "") or getattr(inner, "message", "") or "")


def _chunked(
    items: Sequence[tuple[str, dict[str, Any]]], max_requests: int, max_bytes: int
) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    chunk: list[tuple[str, dict[str, Any]]] = []
    size = 0
    for item in items:
        n = len(json.dumps(item[1], ensure_ascii=False).encode("utf-8"))
        if chunk and (len(chunk) >= max_requests or size + n > max_bytes):
            yield chunk
            chunk, size = [], 0
        chunk.append(item)
        size += n
    if chunk:
        yield chunk


# --------------------------------------------------------------------------- #
# enrichers                                                                    #
# --------------------------------------------------------------------------- #


class ModelEnricher(StageImpl):
    """Base for an enricher that calls a model, in two halves: the requests a
    batch of units needs, and the enrichments made from the answers.

    Two halves so that the same requests can be answered two ways -- now, by
    ``enrich``, concurrently and cache-aware; or ahead of a build and at half
    price, by the pipeline's ``prefill`` through the Message Batches API. One
    code path per prompt, so a batch-built index cannot drift from a live one.

    A unit whose answer failed comes back as ``None`` from ``enrichments_from``;
    ``enrich`` then raises ``EnrichmentIncomplete`` with the rest, and the
    pipeline's ``enrich.on_error`` decides.
    """

    #: The prompt texts. Part of the fingerprint, so an edited prompt misses
    #: the cache even if ``VERSION`` is forgotten.
    PROMPTS: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: Mapping[str, Any],
        *,
        client: Any = None,
        prices: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        super().__init__(params)
        self.claude = Claude(
            client, prices=prices, fallbacks=str(self.param("fallbacks", "default"))
        )

    def requests_for(
        self, units: Sequence[Unit], ctx: EnrichContext
    ) -> list[tuple[str, dict[str, Any]]]:
        """``(key, params)`` per call. Keys are local to this batch of units."""
        raise NotImplementedError

    def enrichments_from(
        self,
        units: Sequence[Unit],
        ctx: EnrichContext,
        answers: Mapping[str, LLMResult | LLMError],
    ) -> list[Enrichment | None]:
        raise NotImplementedError

    def warm_first(self, ctx: EnrichContext) -> bool:
        """Whether this batch's requests share a cached prefix worth warming."""
        return False

    def enrich(self, units: Sequence[Unit], ctx: EnrichContext) -> Sequence[Enrichment]:
        requests = self.requests_for(units, ctx)
        answers = self.claude.call_many(
            [p for _, p in requests],
            max_concurrency=ctx.max_concurrency,
            warm_first=self.warm_first(ctx),
        )
        by_key = {k: a for (k, _), a in zip(requests, answers, strict=True)}
        return complete(self.enrichments_from(units, ctx, by_key), by_key.values())

    def fingerprint(self) -> StageFingerprint:
        return StageFingerprint(
            stage=self.STAGE,
            impl=self.IMPL,
            version=self.VERSION,
            params_hash=hash_obj({"params": self._params, "prompts": list(self.PROMPTS)}),
        )


def complete(
    produced: Sequence[Enrichment | None], answers: Iterable[LLMResult | LLMError]
) -> list[Enrichment]:
    """``produced`` if every unit got an enrichment; otherwise raise
    ``EnrichmentIncomplete`` carrying the ones that did and what failures cost."""
    errors = [a for a in answers if isinstance(a, LLMError)]
    if all(e is not None for e in produced):
        return [e for e in produced if e is not None]
    raise EnrichmentIncomplete(
        produced,
        [f"{e.kind}: {e}" for e in errors] or ["no answer for some units"],
        tokens_in=sum(e.usage.tokens_in for e in errors),
        tokens_out=sum(e.usage.output_tokens for e in errors),
        cost_usd=sum(e.cost_usd for e in errors),
    )


def spent(answer: LLMResult) -> dict[str, Any]:
    """Accounting fields for an ``Enrichment`` made from one answer."""
    return {
        "tokens_in": answer.usage.tokens_in,
        "tokens_out": answer.usage.output_tokens,
        "cost_usd": answer.cost_usd,
    }

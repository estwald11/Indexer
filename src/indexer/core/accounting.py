"""Per-stage cost and latency accounting.

Invariant 6 -- nothing is optimized without a before/after number -- is a
process rule that only holds if the numbers are free to obtain. So accounting is
not an optional wrapper someone remembers to add: every stage call goes through
``Accountant.measure`` and a stage that is not measured does not run.

``StageRun`` is the unit of record for both ingestion and query. It is what the
manifest aggregates and what the ablation table diffs.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

__all__ = ["Accountant", "CacheOutcome", "InMemoryAccountant", "StageFingerprint", "StageRun"]


class CacheOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"
    WRITE = "write"
    BYPASS = "bypass"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class StageFingerprint:
    """Everything about a stage that can change its output.

    The contract on implementations is blunt: **any change that can change
    output must change ``version`` or ``params_hash``**. An enricher whose
    prompt is edited without a version bump will serve stale cache entries
    forever, and the failure looks like the change having no effect -- the worst
    possible symptom, because it reads as evidence against the change.
    """

    stage: str
    impl: str
    version: str
    params_hash: str

    def key(self) -> str:
        return f"{self.stage}/{self.impl}@{self.version}#{self.params_hash[:12]}"

    def as_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "impl": self.impl,
            "version": self.version,
            "params_hash": self.params_hash,
        }


@dataclass(slots=True)
class StageRun:
    """One measured execution of one stage over one input."""

    fingerprint: StageFingerprint
    input_hash: str = ""
    output_hash: str = ""
    wall_ms: float = 0.0
    cache: CacheOutcome = CacheOutcome.DISABLED
    #: Set when a stage was configured off. The run is still recorded, so an
    #: ablation's trace has a row where the stage would have been.
    disabled: bool = False
    items_in: int = 0
    items_out: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.fingerprint.as_dict(),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "wall_ms": round(self.wall_ms, 3),
            "cache": str(self.cache),
            "disabled": self.disabled,
            "items_in": self.items_in,
            "items_out": self.items_out,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error,
            **({"attrs": dict(self.attrs)} if self.attrs else {}),
        }


class Accountant(Protocol):
    """Collects stage runs. One per build or per query."""

    def record(self, run: StageRun) -> None: ...

    @contextmanager
    def measure(self, fingerprint: StageFingerprint) -> Iterator[StageRun]:
        """Time a stage call and record it, including on the exception path.

        Recording failures matters as much as recording successes: a stage that
        fails fast on 30% of documents is cheap and useless, and an accounting
        layer that only sees successes reports it as cheap.
        """
        ...

    def runs(self) -> tuple[StageRun, ...]: ...


class InMemoryAccountant:
    """The default. Holds runs for the life of a build or query.

    A sink that streams to a file or a metrics backend is a drop-in; the seam is
    the ``Accountant`` protocol and nothing else needs to know.
    """

    __slots__ = ("_runs",)

    def __init__(self) -> None:
        self._runs: list[StageRun] = []

    def record(self, run: StageRun) -> None:
        self._runs.append(run)

    @contextmanager
    def measure(self, fingerprint: StageFingerprint) -> Iterator[StageRun]:
        run = StageRun(fingerprint=fingerprint)
        started = time.perf_counter()
        try:
            yield run
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            run.wall_ms = (time.perf_counter() - started) * 1000.0
            self._runs.append(run)

    def runs(self) -> tuple[StageRun, ...]:
        return tuple(self._runs)

    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._runs)

    def by_stage(self) -> dict[str, dict[str, float]]:
        """Aggregate for the manifest: wall, cost and cache rate per stage."""
        agg: dict[str, dict[str, float]] = {}
        for r in self._runs:
            s = agg.setdefault(
                r.fingerprint.stage,
                {
                    "wall_ms": 0.0,
                    "cost_usd": 0.0,
                    "runs": 0.0,
                    "cache_hits": 0.0,
                    "tokens_in": 0.0,
                    "tokens_out": 0.0,
                    "errors": 0.0,
                },
            )
            s["wall_ms"] += r.wall_ms
            s["cost_usd"] += r.cost_usd
            s["runs"] += 1
            s["cache_hits"] += 1 if r.cache is CacheOutcome.HIT else 0
            s["tokens_in"] += r.tokens_in
            s["tokens_out"] += r.tokens_out
            s["errors"] += 1 if r.error else 0
        return agg

"""Retrieval results, ranked lists and the traced response.

Two things are enforced here that are easy to lose:

**Provenance survives ranking.** A ``Hit`` carries its ``Provenance`` directly,
not a unit id to be resolved later. Late resolution is where citations go
missing: the fusion step drops a list, or an index is reconfigured between query
and render, and a passage arrives with nothing to point at.

**Rank fusion needs rank, not just score.** Scores from BM25 and from cosine
similarity are not comparable, and Reciprocal Rank Fusion deliberately does not
try -- it uses position. So ``rank`` is mandatory and 1-based, and ``score``
is explicitly documented as index-local and non-comparable across lists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from indexer.core.ids import DocumentId, UnitId
from indexer.core.provenance import Provenance
from indexer.core.query import Query, RouteDecision
from indexer.core.unit import EnrichedUnit, FieldValue

__all__ = ["Hit", "RankedList", "RecordSet", "RetrievalResponse", "RetrievalStep"]


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved unit, in one ranked list.

    ``score`` is whatever the producing index returned: a BM25 score, a cosine
    similarity, a cross-encoder logit. It is meaningful *within* a list and
    meaningless across lists. Anything comparing scores from different indexes
    without normalising is a bug -- which is why fusion defaults to rank-based.
    """

    unit_id: UnitId
    document_id: DocumentId
    rank: int
    score: float
    index: str
    provenance: Provenance
    #: Hydrated unit, when the index stores enough to return one. Optional so a
    #: vector store that holds only ids and payloads is still a valid index;
    #: the pipeline hydrates from the unit store before returning to a caller.
    unit: EnrichedUnit | None = None
    #: Text as retrieved -- the retrieval surface, which includes any prepended
    #: context. Distinct from the text a caller should *show*, which is
    #: ``unit.unit.text``: showing the LLM-written context as if it were the
    #: document would be a fabricated citation.
    matched_text: str = ""
    #: Which retrieval round produced this, on the iterative path.
    step: int = 0
    explain: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank is 1-based")

    def with_rank(self, rank: int, score: float | None = None) -> Hit:
        return replace(self, rank=rank, score=self.score if score is None else score)


@dataclass(frozen=True, slots=True)
class RankedList:
    """An ordered result list from one source.

    ``source`` is the index name for a retrieval list, and the stage name
    (``"fuse"``, ``"rerank"``) for derived lists, so a trace reads as a chain.
    """

    hits: tuple[Hit, ...]
    source: str
    query_text: str = ""
    #: Impl fingerprint of whatever produced this list, for the manifest.
    fingerprint: str = ""
    latency_ms: float = 0.0
    total_candidates: int | None = None

    def __post_init__(self) -> None:
        for i, h in enumerate(self.hits, start=1):
            if h.rank != i:
                raise ValueError(
                    f"{self.source}: hits must be densely ranked from 1; "
                    f"position {i} claims rank {h.rank}"
                )

    @classmethod
    def from_scored(cls, scored: Sequence[tuple[Hit, float]], source: str, **kw: Any) -> RankedList:
        """Build a correctly-ranked list from unordered (hit, score) pairs."""
        ordered = sorted(scored, key=lambda p: -p[1])
        return cls(
            hits=tuple(h.with_rank(i, s) for i, (h, s) in enumerate(ordered, start=1)),
            source=source,
            **kw,
        )

    def top(self, k: int) -> RankedList:
        return replace(self, hits=self.hits[:k])

    def unit_ids(self) -> tuple[UnitId, ...]:
        return tuple(h.unit_id for h in self.hits)

    def __len__(self) -> int:
        return len(self.hits)


@dataclass(frozen=True, slots=True)
class RecordSet:
    """The structured path's answer: rows, not a ranking.

    ``sources`` keeps the unit ids each row was derived from, so a number in an
    answer is as citable as a passage. A structured store that cannot supply
    them is a documented downgrade, not a silent one.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[FieldValue, ...], ...]
    sources: tuple[tuple[UnitId, ...], ...] = field(default_factory=tuple)
    latency_ms: float = 0.0
    fingerprint: str = ""
    #: Rows the query matched before ``limit`` was applied. ``None`` when the
    #: store cannot say. An agent reading 50 rows must be able to tell "these
    #: are all of them" from "these are the first 50 of 4,000" -- without it,
    #: a truncated list reads as a complete answer and every count derived
    #: from it is silently wrong.
    total: int | None = None
    #: True when ``rows`` is a prefix of a longer answer.
    truncated: bool = False

    def as_dicts(self) -> list[dict[str, FieldValue]]:
        return [dict(zip(self.columns, r, strict=True)) for r in self.rows]

    def is_empty(self) -> bool:
        """Whether the answer contains nothing.

        Not ``not rows``: an aggregate over zero matching rows still returns
        one row (``COUNT = 0``, ``SUM = NULL``), and scoring that as an answer
        rewards a structured path for being asked rather than for finding
        anything. A row with no source unit derives from nothing.
        """
        if not self.rows:
            return True
        if self.sources:
            return not any(self.sources)
        return False


@dataclass(frozen=True, slots=True)
class RetrievalStep:
    """One round of the iterative path, recorded for the trace."""

    step: int
    query_text: str
    lists: tuple[RankedList, ...]
    reason: str = ""
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """What the query pipeline returns. Everything needed to explain itself.

    Deliberately not an "answer": generation is the caller's concern and out of
    scope. What the frame owns is getting the right passages to the caller with
    a trace of how, because invariant 1 says that is where the errors are.
    """

    query: Query
    decision: RouteDecision
    #: Final ranked list after fuse and rerank. Empty on the structured path.
    hits: tuple[Hit, ...] = field(default_factory=tuple)
    #: Populated on the structured path instead of ``hits``.
    records: RecordSet | None = None
    #: Per-index lists before fusion, kept for ablation and debugging.
    retrieved: tuple[RankedList, ...] = field(default_factory=tuple)
    fused: RankedList | None = None
    reranked: RankedList | None = None
    steps: tuple[RetrievalStep, ...] = field(default_factory=tuple)
    #: Stage timings and costs, keyed by stage name.
    latency_ms: Mapping[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    #: Stages that were skipped and why -- an ablation run's primary evidence.
    skipped: Mapping[str, str] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_ms.values())

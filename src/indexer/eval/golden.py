"""The golden-set format.

Built first, because everything else in the brief is a claim that needs a
number, and invariant 1 says the number that matters is a retrieval number.

The one design decision that makes the rest work
------------------------------------------------
**Gold is anchored to document spans, not to unit ids.**

The obvious format records "query X should retrieve unit 7f3a...". It breaks the
moment you change the segmenter, because unit ids are content-derived and every
unit id changes. You then cannot compare a structural segmenter against a
fixed-window one -- which is precisely the comparison the library exists to make
cheap. Worse, it fails *silently*: recall goes to zero and looks like a
catastrophic regression rather than a broken harness.

Anchoring to ``(document_id, span)`` in the parsed document's canonical text
means one golden set survives re-segmentation, re-chunking, re-parsing (as long
as the canonical text is stable) and every ablation arm. A hit counts as
relevant when its span overlaps a gold span sufficiently, under a stated policy.

Graded relevance
----------------
``weight`` is 0-3 in the TREC sense. nDCG needs grades; recall and precision
threshold at ``weight >= 1``. A bootstrapper emits weight 3 for the passage a
question was generated from and 1 for passages a human or judge later marked as
also-sufficient -- the distinction matters because a question with three valid
sources scores badly against a single-gold set for no real reason.

Provenance of the gold itself
-----------------------------
``origin`` records whether an item was machine-generated, human-verified or
mined from logs, and the generator's fingerprint. A bootstrapped set is a
starting point, not a ground truth: it inherits the biases of the model that
wrote it, and any metric computed against it is a self-evaluation until a human
has looked. Recording origin per item keeps that visible, and lets the harness
report verified and unverified subsets separately.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from indexer.core.ids import DocumentId
from indexer.core.provenance import Span
from indexer.core.query import QueryType

__all__ = [
    "GoldOrigin",
    "GoldenQuery",
    "GoldenSet",
    "RelevantSpan",
    "load_golden_set",
    "write_golden_set",
]


class GoldOrigin(StrEnum):
    BOOTSTRAP = "bootstrap"  # machine-generated, unverified
    BOOTSTRAP_VERIFIED = "verified"  # machine-generated, human-checked
    HUMAN = "human"  # human-authored
    PRODUCTION_LOG = "log"  # mined from real traffic, labelled


@dataclass(frozen=True, slots=True)
class RelevantSpan:
    """One passage that should be retrieved for a query.

    ``span`` indexes into ``ParsedDocument.text``, the same coordinate system
    every ``Provenance`` uses -- so scoring is a span-overlap test with no
    id-resolution step that could fail.
    """

    document_id: DocumentId
    span: Span
    weight: int = 3
    #: The text at that span when the item was created. Not used for scoring;
    #: used to *detect* that the corpus moved underneath the golden set, which
    #: otherwise shows up as an unexplained recall drop weeks later.
    snippet: str = ""
    page: int | None = None


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    """One evaluation item."""

    id: str
    query: str
    relevant: tuple[RelevantSpan, ...]
    #: The path the router *should* choose. Scoring the router separately
    #: matters because misrouting is invisible in retrieval metrics: a
    #: structured question sent to vector search fails, and the retrieval
    #: metrics blame the retriever.
    query_type: QueryType = QueryType.FACTUAL
    #: Expected answer, for end-to-end correctness. A string for prose, a
    #: number or date for structured questions -- where exact comparison is
    #: possible and an LLM judge is unnecessary and less reliable.
    answer: str | float | None = None
    answer_type: str = "text"
    origin: GoldOrigin = GoldOrigin.BOOTSTRAP
    generator: str = ""
    difficulty: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def verified(self) -> bool:
        return self.origin in (GoldOrigin.BOOTSTRAP_VERIFIED, GoldOrigin.HUMAN)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["relevant"] = [
            {
                "document_id": r.document_id,
                "span": [r.span.start, r.span.end],
                "weight": r.weight,
                "snippet": r.snippet,
                "page": r.page,
            }
            for r in self.relevant
        ]
        d["query_type"] = str(self.query_type)
        d["origin"] = str(self.origin)
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> GoldenQuery:
        return cls(
            id=d["id"],
            query=d["query"],
            relevant=tuple(
                RelevantSpan(
                    document_id=DocumentId(r["document_id"]),
                    span=Span(int(r["span"][0]), int(r["span"][1])),
                    weight=int(r.get("weight", 3)),
                    snippet=r.get("snippet", ""),
                    page=r.get("page"),
                )
                for r in d["relevant"]
            ),
            query_type=QueryType(d.get("query_type", "factual")),
            answer=d.get("answer"),
            answer_type=d.get("answer_type", "text"),
            origin=GoldOrigin(d.get("origin", "bootstrap")),
            generator=d.get("generator", ""),
            difficulty=d.get("difficulty", ""),
            tags=tuple(d.get("tags", ())),
            notes=d.get("notes", ""),
        )


@dataclass(frozen=True, slots=True)
class GoldenSet:
    """A corpus's evaluation set. JSONL on disk, one ``GoldenQuery`` per line.

    JSONL rather than one JSON document so that a set can be appended to, split
    across shards, diffed line by line in review, and streamed -- all of which
    matter once a set is a living artifact that grows with production traffic.
    """

    queries: tuple[GoldenQuery, ...]
    corpus_id: str = ""
    #: Config hash of the build the spans were taken against. A mismatch at eval
    #: time is a warning, not an error: the spans may still be valid, but
    #: something changed and the run should say so.
    built_against: str = ""
    created_at: str = ""
    notes: str = ""

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[GoldenQuery]:
        return iter(self.queries)

    def verified_only(self) -> GoldenSet:
        from dataclasses import replace

        return replace(self, queries=tuple(q for q in self.queries if q.verified))

    def by_type(self, qt: QueryType) -> GoldenSet:
        from dataclasses import replace

        return replace(self, queries=tuple(q for q in self.queries if q.query_type is qt))

    def stats(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for q in self.queries:
            types[str(q.query_type)] = types.get(str(q.query_type), 0) + 1
        return {
            "total": len(self.queries),
            "verified": sum(1 for q in self.queries if q.verified),
            "by_type": types,
            "mean_relevant_spans": (
                sum(len(q.relevant) for q in self.queries) / len(self.queries)
                if self.queries
                else 0.0
            ),
        }


def load_golden_set(path: str | Path) -> GoldenSet:
    """Read JSONL. A leading ``{"_meta": {...}}`` line carries set-level fields."""
    p = Path(path)
    meta: dict[str, Any] = {}
    items: list[GoldenQuery] = []
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: {exc}") from exc
            if "_meta" in obj:
                meta = obj["_meta"]
                continue
            items.append(GoldenQuery.from_json(obj))
    return GoldenSet(
        queries=tuple(items),
        corpus_id=meta.get("corpus_id", ""),
        built_against=meta.get("built_against", ""),
        created_at=meta.get("created_at", ""),
        notes=meta.get("notes", ""),
    )


def write_golden_set(gs: GoldenSet, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "_meta": {
                        "corpus_id": gs.corpus_id,
                        "built_against": gs.built_against,
                        "created_at": gs.created_at,
                        "notes": gs.notes,
                    }
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for q in gs.queries:
            fh.write(json.dumps(q.to_json(), ensure_ascii=False) + "\n")


def merge(sets: Iterable[GoldenSet]) -> GoldenSet:
    """Combine sets, keeping the most-verified item on an id collision."""
    rank = {
        GoldOrigin.BOOTSTRAP: 0,
        GoldOrigin.PRODUCTION_LOG: 1,
        GoldOrigin.BOOTSTRAP_VERIFIED: 2,
        GoldOrigin.HUMAN: 3,
    }
    best: dict[str, GoldenQuery] = {}
    seen: Sequence[GoldenSet] = list(sets)
    for gs in seen:
        for q in gs.queries:
            cur = best.get(q.id)
            if cur is None or rank[q.origin] > rank[cur.origin]:
                best[q.id] = q
    return GoldenSet(
        queries=tuple(best[k] for k in sorted(best)),
        corpus_id=seen[0].corpus_id if seen else "",
    )

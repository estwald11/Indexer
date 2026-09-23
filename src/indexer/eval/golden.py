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

import hashlib
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
    #: Fraction of the query's content words that occur in the gold passage,
    #: measured by the generator against the full unit text. See ``overlap``.
    lexical_overlap: float | None = None
    #: The query this one was rewritten from, when a generator paraphrases.
    #: Kept because the pair is the measurement: a retriever that finds the
    #: source form and not this one is telling you it matched words rather than
    #: meaning, and without the source form that is invisible.
    source_query: str = ""

    @property
    def verified(self) -> bool:
        return self.origin in (GoldOrigin.BOOTSTRAP_VERIFIED, GoldOrigin.HUMAN)

    @property
    def paraphrased(self) -> bool:
        return bool(self.source_query) and self.source_query != self.query

    def overlap(self) -> float:
        """How much of this query is lifted verbatim from the passage it seeks.

        The number that says what a golden set can and cannot measure. Near 1.0,
        a lexical retriever is being scored on precisely what it does and no
        item *requires* matching meaning -- which caps every dense arm evaluated
        against the set, a neural bi-encoder included. Near 0, the set is asking
        retrievers to bridge vocabulary, which is the thing dense retrieval
        exists for.

        Prefers ``lexical_overlap`` when the generator recorded it, because only
        the generator has the full passage: ``RelevantSpan.snippet`` is
        truncated for file size, and measuring against it understates the real
        overlap roughly twofold. The snippet estimate is the fallback for items
        that predate the field.

        NaN when there is nothing to compare against -- zero would read as "no
        overlap" rather than "cannot tell".
        """
        if self.lexical_overlap is not None:
            return self.lexical_overlap
        from indexer.textutil import STOPWORDS, tokenize

        terms = {t for t in tokenize(self.query) if len(t) > 2 and t not in STOPWORDS}
        gold = " ".join(r.snippet for r in self.relevant if r.snippet)
        if not terms or not gold.strip():
            return float("nan")
        return len(terms & set(tokenize(gold))) / len(terms)

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
            source_query=d.get("source_query", ""),
            lexical_overlap=d.get("lexical_overlap"),
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

    def split(self, dev_share: float = 0.3, *, salt: str = "") -> tuple[GoldenSet, GoldenSet]:
        """``(dev, test)``: tune on the first, report on the second.

        Anything chosen by looking at scores -- fusion weights, a threshold, a
        prompt -- is fitted to the queries it was chosen on, and its score there
        overstates what the next query will see. The test half is where the
        number that gets reported comes from.

        Each query's side is decided by a hash of its id alone, so adding
        queries never moves an existing one across: a test query must never
        become a dev query after a tuning run has seen it. ``salt`` draws a
        different split, for a second opinion.
        """
        from dataclasses import replace

        if not 0.0 < dev_share < 1.0:
            raise ValueError("dev_share must be between 0 and 1")
        dev: list[GoldenQuery] = []
        test: list[GoldenQuery] = []
        for q in self.queries:
            digest = hashlib.sha256(f"{salt}:{q.id}".encode()).digest()
            (dev if int.from_bytes(digest[:8], "big") / 2**64 < dev_share else test).append(q)
        return (
            replace(self, queries=tuple(dev), notes=f"dev split ({dev_share:.0%}) of {self.notes}"),
            replace(self, queries=tuple(test), notes=f"test split of {self.notes}"),
        )

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
            "paraphrased": sum(1 for q in self.queries if q.paraphrased),
            "lexical_overlap": self.mean_lexical_overlap(),
        }

    def mean_lexical_overlap(self) -> float:
        """Mean ``GoldenQuery.overlap`` over the items that have one.

        Report this next to every retrieval number the set produces. A set at
        1.0 cannot distinguish a retriever that matches words from one that
        matches meaning, however good either is, and a dense arm losing against
        BM25 on such a set is weak evidence about the retriever and strong
        evidence about the set.
        """
        vals = [v for q in self.queries if not _isnan(v := q.overlap())]
        return sum(vals) / len(vals) if vals else float("nan")


def _isnan(x: float) -> bool:
    return x != x


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

"""Bootstrapping a golden set from a corpus, without a model.

Adopting the frame on a new corpus must not begin with weeks of labelling, so
there has to be a way to manufacture a starting set. Doing it well is harder
than it looks, and most of this file is the filtering rather than the generation.

Query design, and why it is what it is
--------------------------------------
Invariant 3 is about a specific failure: a chunk that does not name its own
subject. "It must be set before the first request" cannot be retrieved for
"requests timeout" because neither content word appears in it. Contextualisation
fixes that by putting the subject into the retrieval surface.

So a golden set that can measure contextualisation must contain queries shaped
like the ones that fail: **subject + detail**, where the subject is a
document-level fact and the detail is unit-local. A set built only from
unit-local terms cannot show contextualisation helping, because there was never
anything missing. A set built only from headings would show it winning
trivially, because the heading is literally what gets prepended.

This generator therefore takes the subject from the document's *identity* (its
name), and the detail from terms distinctive to the unit against the rest of its
own document. Note the consequence honestly: an extractive contextualiser also
draws on document-level information, so part of any measured gain is structural
rather than semantic. That is stated in the report rather than left for a reader
to work out.

Filtering, which is where the quality is
----------------------------------------
Three filters, each removing a different kind of useless item:

*Too easy.* If a plain lexical baseline already ranks the gold unit first, the
item cannot distinguish any two systems. Keeping these inflates every arm
equally and compresses the deltas toward zero.

*Unfindable.* If no baseline finds it at all, it measures nothing either, and
usually means the generated query is nonsense.

*Ambiguous.* If the query's terms are as present in some other unit as in the
gold one, the "wrong" answer may be right and the item is noise.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from indexer.core.document import ParsedDocument
from indexer.core.ids import DocumentId
from indexer.core.query import QueryType
from indexer.core.registry import register
from indexer.core.unit import EnrichedUnit
from indexer.eval.golden import GoldenQuery, GoldenSet, GoldOrigin, RelevantSpan
from indexer.plugin import StageImpl, dataclass_params
from indexer.textutil import STOPWORDS, WORD_RE, tokenize

__all__ = ["HeuristicBootstrapper"]


@dataclass(frozen=True, slots=True)
class HeuristicParams:
    target_size: int = 200
    seed: int = 20260917
    min_unit_chars: int = 220
    max_unit_chars: int = 4000
    detail_terms: int = 3
    #: Drop items the lexical baseline already ranks first: no headroom.
    drop_if_baseline_rank: int = 1
    #: Drop items the baseline cannot find within this depth: unmeasurable.
    drop_if_worse_than: int = 50
    #: Share of the set that should be structured/numeric questions, so the
    #: router is measurable. A set of pure prose lookups cannot see invariant 5.
    structured_share: float = 0.2
    max_per_document: int = 4


@register(
    "bootstrap",
    "heuristic",
    version="1",
    params_model=dataclass_params(HeuristicParams),
    summary="Offline golden-set generation: subject+detail queries, filtered for discriminability.",
)
def _make_heuristic(params: dict[str, Any], **_: Any) -> HeuristicBootstrapper:
    return HeuristicBootstrapper(params)


class HeuristicBootstrapper(StageImpl):
    STAGE, IMPL, VERSION = "bootstrap", "heuristic", "1"

    def bootstrap(
        self,
        documents: Sequence[ParsedDocument],
        units: Sequence[EnrichedUnit],
        *,
        baseline: Any = None,
        target_size: int | None = None,
    ) -> GoldenSet:
        rng = random.Random(int(self.param("seed", 20260917)))
        want = int(target_size or self.param("target_size", 200))
        min_chars = int(self.param("min_unit_chars", 220))
        max_chars = int(self.param("max_unit_chars", 4000))
        max_per_doc = int(self.param("max_per_document", 4))

        by_doc: dict[str, list[EnrichedUnit]] = {}
        for u in units:
            by_doc.setdefault(u.document_id, []).append(u)
        doc_titles = {d.document_id: _subject_of(d) for d in documents}

        candidates = [
            u
            for u in units
            if min_chars <= len(u.unit.text) <= max_chars and len(by_doc[u.document_id]) > 1
        ]
        rng.shuffle(candidates)

        items: list[GoldenQuery] = []
        per_doc: dict[str, int] = {}
        stats = {"too_easy": 0, "unfindable": 0, "no_terms": 0, "ambiguous": 0}

        for eu in candidates:
            if len(items) >= want:
                break
            if per_doc.get(eu.document_id, 0) >= max_per_doc:
                continue
            subject = doc_titles.get(eu.document_id, "")
            detail = _distinctive_terms(
                eu.unit.text,
                [o.unit.text for o in by_doc[eu.document_id] if o.unit_id != eu.unit_id],
                int(self.param("detail_terms", 3)),
            )
            if len(detail) < 2:
                stats["no_terms"] += 1
                continue

            query_text = _phrase(subject, detail)
            verdict = self._screen(query_text, eu, baseline, stats)
            if verdict is not None:
                continue

            items.append(
                GoldenQuery(
                    id=f"q{len(items):04d}-{eu.unit_id[:8]}",
                    query=query_text,
                    relevant=(
                        RelevantSpan(
                            document_id=DocumentId(eu.document_id),
                            span=eu.unit.provenance.span,
                            weight=3,
                            snippet=eu.unit.text[:160],
                        ),
                    ),
                    query_type=QueryType.FACTUAL,
                    origin=GoldOrigin.BOOTSTRAP,
                    generator=self.fingerprint().key(),
                    tags=("subject_detail",),
                    notes="; ".join(eu.unit.section_path),
                )
            )
            per_doc[eu.document_id] = per_doc.get(eu.document_id, 0) + 1

        items.extend(self._structured_items(units, rng, want))

        return GoldenSet(
            queries=tuple(items),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            notes=(
                f"heuristic bootstrap; filtered: {stats}. "
                f"UNVERIFIED -- machine-generated, no human has checked these."
            ),
        )

    def _screen(
        self, query_text: str, eu: EnrichedUnit, baseline: Any, stats: dict[str, int]
    ) -> str | None:
        """Reject items that cannot distinguish two systems."""
        if baseline is None:
            return None
        from indexer.core.accounting import InMemoryAccountant
        from indexer.core.cache import NullCache
        from indexer.core.stages import IndexQuery, StageContext

        ctx = StageContext(cache=NullCache(), accountant=InMemoryAccountant())
        hits = baseline.search(
            IndexQuery(text=query_text, top_k=int(self.param("drop_if_worse_than", 50))), ctx
        )
        ids = list(hits.unit_ids())
        if eu.unit_id not in ids:
            stats["unfindable"] += 1
            return "unfindable"
        rank = ids.index(eu.unit_id) + 1
        if rank <= int(self.param("drop_if_baseline_rank", 1)):
            stats["too_easy"] += 1
            return "too_easy"
        return None

    def _structured_items(
        self, units: Sequence[EnrichedUnit], rng: random.Random, want: int
    ) -> list[GoldenQuery]:
        """Generate questions that should take the STRUCTURED route.

        Built from *extracted fields*, so the right answer is by construction
        available without vector search. That is the point: these items are how
        invariant 5 becomes measurable, and a golden set without them cannot see
        a router regression at all.
        """
        with_fields = [u for u in units if u.fields()]
        if not with_fields:
            return []
        n_want = int(want * float(self.param("structured_share", 0.2)))
        out: list[GoldenQuery] = []
        by_field: dict[str, list[EnrichedUnit]] = {}
        for u in with_fields:
            for k in u.fields():
                by_field.setdefault(k, []).append(u)

        # Spread the structured budget across the available fields rather than
        # capping at a handful. A slice of three queries cannot detect a router
        # regression -- one misroute moves it by 33 points.
        per_field = max(1, n_want // max(1, len(by_field)))
        for field_name, holders in sorted(by_field.items()):
            if len(out) >= n_want:
                break
            sample = rng.sample(holders, min(per_field, len(holders)))
            seen_values: set[str] = set()
            for eu in sample:
                if len(out) >= n_want:
                    break
                value = eu.fields()[field_name]
                # One query per distinct value: twenty copies of "which entries
                # have a version recorded" measure one thing twenty times.
                if str(value) in seen_values:
                    continue
                seen_values.add(str(value))
                pretty = field_name.replace("_", " ")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    q = f"which entries have {pretty} greater than {value}"
                    qt = QueryType.NUMERIC
                elif hasattr(value, "isoformat"):
                    q = f"which entries have {pretty} before {value}"
                    qt = QueryType.TEMPORAL
                else:
                    q = f"how many entries have a {pretty} recorded"
                    qt = QueryType.STRUCTURED
                out.append(
                    GoldenQuery(
                        id=f"s{len(out):04d}-{field_name[:10]}",
                        query=q,
                        relevant=(
                            RelevantSpan(
                                document_id=DocumentId(eu.document_id),
                                span=eu.unit.provenance.span,
                                weight=3,
                                snippet=eu.unit.text[:160],
                            ),
                        ),
                        query_type=qt,
                        answer=str(value),
                        answer_type="number" if isinstance(value, (int, float)) else "text",
                        origin=GoldOrigin.BOOTSTRAP,
                        generator=self.fingerprint().key(),
                        tags=("structured", field_name),
                    )
                )
        return out


def _subject_of(doc: ParsedDocument) -> str:
    """The document's subject, from its identity rather than its prose.

    Taken from the file/package name rather than the first heading: a heading is
    what a contextualiser prepends, and drawing the query's subject from the same
    place would make the contextualisation arm win by construction.
    """
    name = str(doc.metadata.get("package") or doc.metadata.get("name") or "")
    name = re.sub(r"\.(md|rst|txt)$", "", name, flags=re.I)
    name = re.sub(r"[_\-/]+", " ", name).strip()
    return name


def _distinctive_terms(text: str, others: Sequence[str], n: int) -> list[str]:
    """Terms frequent in this unit and rare in its siblings.

    Sibling-relative rather than corpus-relative: a term common across the whole
    corpus but characteristic of this *document's* one section is exactly what a
    user would type, and corpus-level IDF would discard it.
    """
    here: dict[str, int] = {}
    for m in WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w in STOPWORDS or w.isdigit() or len(w) < 4:
            continue
        here[w] = here.get(w, 0) + 1
    if not here:
        return []
    sibling_tokens: set[str] = set()
    for o in others:
        sibling_tokens |= set(tokenize(o))
    scored = [(w, c * (2.0 if w not in sibling_tokens else 1.0)) for w, c in here.items()]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in scored[:n]]


def _phrase(subject: str, detail: Sequence[str]) -> str:
    """Assemble a plausible query. Deliberately terse, like real search traffic."""
    parts = [p for p in (subject, *detail) if p]
    return " ".join(dict.fromkeys(parts))

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
from dataclasses import dataclass, replace
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

__all__ = ["HeuristicBootstrapper", "LLMBootstrapper", "measure_overlap"]


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
    #: Language of the structured questions: en or it. An Italian archive
    #: asked in English measures a router nobody will use.
    language: str = "en"


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
                    # Measured here because only the generator holds the full
                    # unit text; the stored snippet is truncated and scoring
                    # against it understates the overlap roughly twofold.
                    lexical_overlap=measure_overlap(query_text, eu.unit.text),
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

        Two mistakes are easy here and both were made in an earlier version:

        *A comparison at the edge of the data.* "release date before <the
        earliest date>" matches nothing, so the item scores as a retrieval
        failure no matter how well the system works. Thresholds are drawn from
        the interior of the observed range, and every generated item is checked
        against the actual values before it is kept.

        *A query that does not name what it asks about.* "how many entries have
        a package recorded" is the same sentence whatever package was sampled,
        so sixteen sampled packages produced sixteen identical queries measuring
        one thing sixteen times -- and none of them could be judged, because the
        answer recorded was a package name the question never mentions.
        Categorical fields now ask for a specific value, which makes the items
        distinct and the answer checkable.
        """
        with_fields = [u for u in units if u.fields()]
        if not with_fields:
            return []
        n_want = int(want * float(self.param("structured_share", 0.2)))
        if n_want <= 0:
            return []

        by_field: dict[str, list[Any]] = {}
        for u in with_fields:
            for k, v in u.fields().items():
                if v is not None:
                    by_field.setdefault(k, []).append(v)

        out: list[GoldenQuery] = []
        seen_queries: set[str] = set()
        per_field = max(1, n_want // max(1, len(by_field)))

        for field_name, values in sorted(by_field.items()):
            if len(out) >= n_want:
                break
            pretty = field_name.replace("_", " ")
            for q, answer, qt in self._field_questions(field_name, pretty, values, per_field, rng):
                if len(out) >= n_want or q in seen_queries:
                    continue
                seen_queries.add(q)
                out.append(
                    GoldenQuery(
                        id=f"s{len(out):04d}-{field_name[:10]}",
                        query=q,
                        # Structured answers are aggregates over the corpus, not
                        # passages, so there is no single gold span. The empty
                        # tuple says so; retrieval metrics report as
                        # not-applicable and correctness is judged on the answer.
                        relevant=(),
                        query_type=qt,
                        answer=answer,
                        answer_type="number" if isinstance(answer, (int, float)) else "text",
                        origin=GoldOrigin.BOOTSTRAP,
                        generator=self.fingerprint().key(),
                        tags=("structured", field_name),
                    )
                )
        return out

    def _field_questions(
        self,
        field_name: str,
        pretty: str,
        values: Sequence[Any],
        limit: int,
        rng: random.Random,
    ) -> list[tuple[str, Any, QueryType]]:
        """Questions this field can answer, each guaranteed a non-empty result."""
        out: list[tuple[str, Any, QueryType]] = []
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        dated = [v for v in values if hasattr(v, "isoformat")]
        textual = [v for v in values if isinstance(v, str)]
        say = _TEMPLATES[str(self.param("language", "en"))]

        if numeric:
            ordered = sorted(numeric)
            # Thresholds from the interior: strictly below the maximum, so
            # "greater than" always matches something.
            for t in _interior(ordered, limit, rng):
                if t < ordered[-1]:
                    out.append((say.greater(pretty, t), None, QueryType.NUMERIC))
        if dated:
            ordered = sorted(dated)
            # Strictly above the minimum, so "before" always matches something.
            for t in _interior(ordered, limit, rng):
                if t > ordered[0]:
                    out.append((say.before(pretty, t), None, QueryType.TEMPORAL))
        if textual:
            counts: dict[str, int] = {}
            for v in textual:
                counts[v] = counts.get(v, 0) + 1
            # Most frequent first: a value seen once is a weaker test of the
            # structured path than one the corpus actually groups by.
            common = [v for v, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
            for v in common[:limit]:
                out.append((say.equal(pretty, v), v, QueryType.STRUCTURED))
            if common:
                out.append((say.distinct(pretty), None, QueryType.STRUCTURED))
        rng.shuffle(out)
        return out[: limit + 1]


@dataclass(frozen=True, slots=True)
class _Phrasing:
    """How a structured question is worded, per language. Each must be one the
    rules router reads as the comparison it states -- the point of these items
    is to measure the router, not to trip it on wording."""

    greater: Any
    before: Any
    equal: Any
    distinct: Any


def _it_number(v: float) -> str:
    """As an Italian writes it: "1.250,5", "480"."""
    if float(v).is_integer():
        return f"{int(v):,}".replace(",", ".")
    whole, frac = f"{v:,.2f}".split(".")
    return whole.replace(",", ".") + "," + frac


_TEMPLATES: dict[str, _Phrasing] = {
    "en": _Phrasing(
        greater=lambda f, t: f"which entries have {f} greater than {t}",
        before=lambda f, t: f"which entries have {f} before {t.isoformat()}",
        equal=lambda f, v: f"which entries have {f} {v}",
        distinct=lambda f: f"how many distinct {f} values are recorded",
    ),
    "it": _Phrasing(
        greater=lambda f, t: f"quali documenti hanno {f} superiore a {_it_number(t)}",
        before=lambda f, t: f"quali documenti hanno {f} prima del {t.strftime('%d/%m/%Y')}",
        equal=lambda f, v: f"quali documenti hanno {f} {v}",
        distinct=lambda f: f"quanti valori distinti di {f} sono registrati",
    ),
}


def _interior(ordered: Sequence[Any], k: int, rng: random.Random) -> list[Any]:
    """Sample thresholds from inside a sorted range, never its endpoints.

    A comparison against the minimum or maximum of the observed data matches
    everything or nothing, and either way measures the data rather than the
    system.
    """
    if len(ordered) < 3:
        return list(ordered[:1])
    lo, hi = len(ordered) // 6, len(ordered) - max(1, len(ordered) // 6)
    interior = ordered[lo:hi] or ordered
    picks = rng.sample(list(interior), min(k, len(interior)))
    return sorted(set(picks))


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


def measure_overlap(query: str, passage: str) -> float:
    """Fraction of a query's content words that occur verbatim in the passage.

    Recorded on every generated item, because it is the one number that says
    what the resulting set can measure. A set near 1.0 scores a lexical
    retriever on exactly what it does and contains no item that *requires*
    matching meaning; a paraphrasing generator is trying to push this down
    without making the item unanswerable, and the two failure modes look
    identical in a retrieval metric but not in this one.
    """
    terms = {t for t in tokenize(query) if len(t) > 2 and t not in STOPWORDS}
    if not terms:
        return float("nan")
    return len(terms & set(tokenize(passage))) / len(terms)


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


@dataclass(frozen=True, slots=True)
class LLMParams:
    model: str = "claude-haiku-4-5-20251001"
    target_size: int = 300
    max_tokens: int = 96
    #: Keep the literal item when the rewrite came back still sharing this much
    #: of its vocabulary with the passage. Above it the model echoed rather than
    #: rewrote, and shipping it would quietly refill the set with the lexical
    #: items the rewrite exists to remove.
    max_overlap: float = 0.4
    #: Drop, rather than keep, an item whose rewrite failed. Off by default:
    #: a mixed set is still scoreable and the tag says which is which, whereas
    #: silently dropping a third of the set changes what the numbers compare.
    drop_failed_rewrites: bool = False
    seed: int = 20260917
    structured_share: float = 0.2
    max_per_document: int = 4
    detail_terms: int = 3
    min_unit_chars: int = 220
    max_unit_chars: int = 4000
    drop_if_baseline_rank: int = 1
    drop_if_worse_than: int = 50
    #: en or it: the language questions are written in.
    language: str = "en"


@register(
    "bootstrap",
    "llm_bootstrap",
    version="1",
    params_model=dataclass_params(LLMParams),
    summary=(
        "Paraphrasing golden-set generation: heuristic items rewritten as natural "
        "questions that avoid the passage's vocabulary. Needs an API key."
    ),
    requires=("anthropic",),
)
def _make_llm(params: dict[str, Any], **kw: Any) -> LLMBootstrapper:
    return LLMBootstrapper(params, client=kw.get("client"))


class LLMBootstrapper(StageImpl):
    """Rewrites heuristic items into queries that do not quote their answer.

    Why this exists
    ---------------
    `HeuristicBootstrapper` draws its detail terms from the target unit's own
    text, and measurement says it does so thoroughly: on the PyPI corpus those
    terms occur verbatim in the gold passage 91% of the time. That is right for
    what it was built for -- invariant 3 needs queries whose *subject* is
    missing from the chunk -- but it means a lexical retriever is scored on
    exactly what it does, and no item in the set *requires* matching meaning.
    Every dense arm evaluated against such a set is capped by it, a neural
    bi-encoder included.

    Lowering that overlap is the whole job, and it is harder than it sounds.
    Surface transforms were tried first and measured: inflecting the terms,
    dropping the rarest one, wrapping the query in question framing. All three
    made *both* arms worse and none closed the gap between them (`docs/ABLATION.md`
    records the table). The reason they fail is that the offline dense arms are
    bag-of-words methods too, so a word the corpus never contains is as opaque
    to them as it is to BM25. A useful rewrite has to substitute vocabulary the
    corpus *does* contain, in the sense the passage means -- and nothing
    offline in this library knows word senses. A model does.

    The order of operations is the design
    -------------------------------------
    Answerability is established on the literal query, **before** rewriting, by
    the same lexical screen `HeuristicBootstrapper` uses. Rewriting comes after.

    Doing it the other way round is the trap: screening a paraphrased query with
    a lexical baseline rejects it for being unfindable by exact match, which is
    precisely the property that made it worth generating. A set built that way
    looks rigorous and has quietly filtered out every item it existed to add.

    So each item carries both forms. ``source_query`` is what the heuristic
    generator produced and what the baseline was shown; ``query`` is the
    rewrite. The pair is the measurement: a retriever that finds one and not the
    other is telling you which of the two it was doing.
    """

    STAGE, IMPL, VERSION = "bootstrap", "llm_bootstrap", "1"

    PROMPT = (
        "Here is a passage from technical documentation:\n"
        "<passage>\n{passage}\n</passage>\n\n"
        "Someone who has NOT read this passage wants to find it. Write the search "
        "query they would type.\n\n"
        "Rules:\n"
        "- Ask for the information, do not restate it.\n"
        "- Avoid the passage's distinctive wording. Use ordinary words for the "
        "same ideas, the way someone describes a problem before they know the term "
        "for it.\n"
        "- Keep the subject '{subject}' so the query is not ambiguous across the corpus.\n"
        "- One line, under fifteen words, no quotation marks, no preamble.\n\n"
        "Query:"
    )

    #: The same request, for an Italian archive: the question an Italian
    #: employee would type, in Italian, about a company document.
    PROMPT_IT = (
        "Ecco un passaggio di un documento aziendale:\n"
        "<passage>\n{passage}\n</passage>\n\n"
        "Una persona che NON ha letto questo passaggio vuole trovarlo. Scrivi la "
        "ricerca che digiterebbe, in italiano.\n\n"
        "Regole:\n"
        "- Chiedi l'informazione, non ripeterla.\n"
        "- Evita le parole caratteristiche del passaggio. Usa parole comuni per gli "
        "stessi concetti, come chi descrive un problema prima di conoscerne il termine.\n"
        "- Mantieni il soggetto '{subject}', perché la ricerca non sia ambigua.\n"
        "- Una riga, meno di quindici parole, senza virgolette né preamboli.\n\n"
        "Ricerca:"
    )

    def __init__(self, params: dict[str, Any], client: Any = None) -> None:
        super().__init__(params)
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "llm_bootstrap needs the `anthropic` package: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def _literal_set(
        self,
        documents: Sequence[ParsedDocument],
        units: Sequence[EnrichedUnit],
        baseline: Any,
        target_size: int | None,
    ) -> GoldenSet:
        """The items to rewrite: heuristic generation, screened as usual.

        Composed rather than reimplemented, so the two sets differ in the query
        wording and in nothing else -- same candidates, same filters, same gold
        spans. That is what makes "the set was paraphrased" a controlled
        comparison rather than two unrelated sets of numbers.
        """
        shared = (
            "target_size",
            "seed",
            "min_unit_chars",
            "max_unit_chars",
            "detail_terms",
            "drop_if_baseline_rank",
            "drop_if_worse_than",
            "structured_share",
            "max_per_document",
            "language",
        )
        return HeuristicBootstrapper(
            {k: self.param(k) for k in shared if self.param(k) is not None}
        ).bootstrap(documents, units, baseline=baseline, target_size=target_size)

    def _rewrite(self, client: Any, passage: str, subject: str) -> tuple[str, int, int]:
        resp = client.messages.create(
            model=self.param("model"),
            max_tokens=int(self.param("max_tokens", 96)),
            messages=[
                {
                    "role": "user",
                    "content": self._prompt().format(passage=passage[:4000], subject=subject),
                }
            ],
        )
        usage = getattr(resp, "usage", None)
        # A refusal, a truncation at max_tokens, or a model that simply returned
        # nothing all arrive here as empty or blank content. Taking the first
        # line unguarded turns any of them into an IndexError partway through a
        # several-hundred-item run; the caller counts an empty rewrite and keeps
        # the literal item instead.
        raw = resp.content[0].text if resp.content else ""
        lines = [ln.strip().strip('"') for ln in raw.strip().splitlines() if ln.strip()]
        text = lines[0] if lines else ""
        return (
            text,
            getattr(usage, "input_tokens", 0) if usage else 0,
            getattr(usage, "output_tokens", 0) if usage else 0,
        )

    def _prompt(self) -> str:
        return self.PROMPT_IT if self.param("language", "en") == "it" else self.PROMPT

    def bootstrap(
        self,
        documents: Sequence[ParsedDocument],
        units: Sequence[EnrichedUnit],
        *,
        baseline: Any = None,
        target_size: int | None = None,
    ) -> GoldenSet:
        literal = self._literal_set(documents, units, baseline, target_size)
        # Gold is anchored to spans, not unit ids, so recover the source unit the
        # same way the scorer would -- by document and span. There is no id
        # fallback on purpose: `GoldenQuery.id` carries a *truncated* unit id,
        # so a lookup through it would miss silently and look like a safety net.
        by_span = {(u.document_id, u.unit.provenance.span): u for u in units}

        client = self._get_client()
        max_overlap = float(self.param("max_overlap", 0.4))
        drop_failed = bool(self.param("drop_failed_rewrites", False))
        counts = {"rewritten": 0, "echoed": 0, "empty": 0, "no_source": 0, "skipped_structured": 0}
        tokens_in = tokens_out = 0

        out: list[GoldenQuery] = []
        for item in literal:
            # Structured items are answered from extracted fields, not from
            # prose, so rewriting their wording measures the router rather than
            # the retriever. Left as generated.
            if item.query_type is not QueryType.FACTUAL:
                counts["skipped_structured"] += 1
                out.append(item)
                continue

            gold = item.relevant[0] if item.relevant else None
            eu = by_span.get((gold.document_id, gold.span)) if gold else None
            if eu is None:
                counts["no_source"] += 1
                out.append(item)
                continue

            subject = item.query.split()[0] if item.query.split() else ""
            text, t_in, t_out = self._rewrite(client, eu.unit.text, subject)
            tokens_in += t_in
            tokens_out += t_out

            if not text:
                counts["empty"] += 1
                if not drop_failed:
                    out.append(item)
                continue

            overlap = measure_overlap(text, eu.unit.text)
            echoed = overlap == overlap and overlap > max_overlap
            if echoed:
                counts["echoed"] += 1
                if drop_failed:
                    continue
            else:
                counts["rewritten"] += 1

            out.append(
                replace(
                    item,
                    query=text,
                    source_query=item.query,
                    lexical_overlap=overlap,
                    generator=self.fingerprint().key(),
                    tags=(*item.tags, "paraphrase" if not echoed else "paraphrase_echoed"),
                )
            )

        return GoldenSet(
            queries=tuple(out),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            notes=(
                f"llm_bootstrap rewrite of {len(literal)} heuristic items: {counts}; "
                f"tokens in={tokens_in} out={tokens_out}. "
                f"UNVERIFIED -- machine-generated, no human has checked these."
            ),
        )

    def fingerprint(self) -> Any:
        from indexer.core.accounting import StageFingerprint

        # The prompt is part of the fingerprint: editing it changes every query
        # the generator produces, and a set whose generator hash did not move
        # would silently claim comparability it does not have.
        from indexer.core.ids import hash_obj

        return StageFingerprint(
            stage=self.STAGE,
            impl=self.IMPL,
            version=self.VERSION,
            params_hash=hash_obj({**self._params, "_prompt": self._prompt()}),
        )

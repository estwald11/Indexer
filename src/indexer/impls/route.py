"""Routers: rules-based, and a passthrough.

The rules router is deliberately the reference implementation rather than a
placeholder for an LLM one. Reasons, in order of weight:

1. It is ~0 latency and ~0 cost on the path that carries most traffic.
2. It is the control arm. "The LLM router is worth it" is a claim that needs a
   cheap baseline to be measured against, and a rules router is a strong one --
   the signals that distinguish a numeric question from a prose one are mostly
   lexical.
3. It is deterministic, so a routing regression is attributable to a rule rather
   than to sampling.

Every decision it makes is logged with the rule that fired, which is what makes
"40% of traffic is taking the iterative path" a diagnosable observation rather
than a mystery.

Italian
-------
The rules used to be English words only. "fatture con importo superiore a 1000
euro" matched no rule, was classified as prose, and went to vector search -- the
exact failure invariant 5 exists to prevent, and invisible in aggregate metrics.
The patterns, operators and fillers are now bilingual, numbers and dates are
read as the query's locale writes them ("1.000,50", "30/06/2025", "giugno
2025", "nel 2024"), and three vocabularies a deployment configures teach the
router its archive's words without code:

``field_aliases``   {"data_documento": ["data", "emessa", "emesse"]}
``value_aliases``   {"tipo_documento": {"fattura": ["fattura", "fatture"]}}
``default_date_field`` / ``default_measure``
                    where "nel 2024" and "sopra i 1000 euro" apply when the
                    question names no field.

It also asks for aggregates when the question does -- "quante", "how many",
"totale", "media", "per fornitore" -- instead of listing rows, and at the
configured ``level`` (``document`` counts invoices, ``unit`` counts chunks).
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from indexer.analysis import fold_accents, italian_light_stem
from indexer.core.predicate import (
    Aggregation,
    AggregationOp,
    And,
    Compare,
    Exists,
    Op,
    Predicate,
    StructuredQuery,
    TextMatch,
    all_of,
)
from indexer.core.query import Query, QueryType, RouteDecision, RoutePath, RouteTarget
from indexer.core.registry import register
from indexer.core.stages import StageContext
from indexer.normalize import parse_date, parse_number
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["PassthroughRouter", "RulesRouter"]

# Ordered most-specific first: the first rule that fires wins, and an aggregate
# question that also mentions a date is an aggregate question.
_AGGREGATE = re.compile(
    r"\b(how many|count of|number of|total|sum of|average|mean|median|"
    r"most|least|largest|smallest|highest|lowest|top \d+|list all|which \w+ (have|has|are)|"
    r"quant[ie]|quanto|numero (di|delle|dei|degli)|conta\w*|conteggio|totale|somma|media|"
    r"medio|massim[oa]|minim[oa]|elenc\w+|qual[ie] \w+ (hanno|ha|sono))\b",
    re.I,
)
_TEMPORAL = re.compile(
    r"\b(before|after|since|until|between .{1,20} and|earlier than|later than|"
    r"in \d{4}|as of|expir\w+|effective|released? (in|on)|last (year|month|week)|"
    r"prima (del|della|dello|dell|di)|dopo (il|la|lo|l|del|della|dell)|dal|dalla|"
    r"fino (al|alla|a)|entro|tra .{1,20} e|nel \d{4}|nell anno|scad\w+|successiv\w+|"
    r"precedent\w+|anterior\w+|posterior\w+|a partire da\w*|ultim[oa] (anno|mese|trimestre))\b",
    re.I,
)
_NUMERIC = re.compile(
    r"(\b(greater|less|more|fewer|at least|at most|over|under|above|below)\s+than?\b"
    r"|[<>]=?\s*\d|\b\d+(\.\d+)?\s*(%|usd|eur|gbp|k|m|bn|million|billion)\b"
    r"|\b(superior|inferior|maggior|minor)[ei]\s+(a|di|ai|al)\b|\bpiù di\b|\bmeno di\b"
    r"|\boltre\b|\balmeno\b|\bal massimo\b|\bnon (più|meno) di\b|\bpari a\b|\buguale a\b"
    r"|\bsopra (i|a)\b|\bsotto (i|a)\b|[€$]\s*\d|\d[\d.,]*\s*(euro|€|mila|milion\w*)\b)",
    re.I,
)
_COMPARATIVE = re.compile(
    r"\b(compare|versus|vs\.?|difference between|better than|instead of|"
    r"trade-?offs? between|confront\w+|rispetto a|differenz\w+ tra|meglio di|peggio di|"
    r"anziché|invece di)\b",
    re.I,
)
_MULTI_HOP = re.compile(
    r"\b(and (then|also)|as well as|both .{1,30} and|why does .{1,40} when|"
    r"how does .{1,30} affect|what causes|implications? of|e poi|oltre a|"
    r"sia .{1,30} che|perché .{1,40} quando|come influisc\w+|cosa causa|implicazion\w+)\b",
    re.I,
)
_SUMMARY = re.compile(
    r"\b(summar\w+|overview of|explain|what is the purpose|walk me through|riassum\w+|"
    r"riassunto|riepilog\w+|sintesi|panoramica|spiega\w*|qual è lo scopo|illustra\w*)\b",
    re.I,
)
_MONTH_NAMES = (
    r"gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|"
    r"dicembre|january|february|march|april|may|june|july|august|september|october|"
    r"november|december"
)
_DATE = re.compile(
    rf"\b(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{2,4}}|\d{{4}}|{_MONTH_NAMES})\b",
    re.I,
)

#: Comparison words, longest first so "greater than or equal" beats "greater".
#: Temporal words are comparisons too -- without them every temporal question
#: collapses to equality and matches nothing.
_OPERATORS: dict[str, Op] = {
    "greater than or equal to": Op.GTE,
    "less than or equal to": Op.LTE,
    "at least": Op.GTE,
    "no less than": Op.GTE,
    "at most": Op.LTE,
    "no more than": Op.LTE,
    "greater than": Op.GT,
    "more than": Op.GT,
    "larger than": Op.GT,
    "later than": Op.GT,
    "less than": Op.LT,
    "fewer than": Op.LT,
    "smaller than": Op.LT,
    "earlier than": Op.LT,
    "not after": Op.LTE,
    "not before": Op.GTE,
    "before": Op.LT,
    "after": Op.GT,
    "since": Op.GTE,
    "until": Op.LTE,
    "above": Op.GT,
    "below": Op.LT,
    "over": Op.GT,
    "under": Op.LT,
    ">=": Op.GTE,
    "<=": Op.LTE,
    ">": Op.GT,
    "<": Op.LT,
    "=": Op.EQ,
    # Italian. Articles are folded in ("superiore ai", "dopo il") because they
    # sit between the operator and the value in every real question.
    "maggiore o uguale a": Op.GTE,
    "maggiore o uguale di": Op.GTE,
    "minore o uguale a": Op.LTE,
    "minore o uguale di": Op.LTE,
    "superiore o uguale a": Op.GTE,
    "inferiore o uguale a": Op.LTE,
    "non inferiore a": Op.GTE,
    "non superiore a": Op.LTE,
    "non meno di": Op.GTE,
    "non più di": Op.LTE,
    "maggiore di": Op.GT,
    "maggiore a": Op.GT,
    "maggiori di": Op.GT,
    "superiore a": Op.GT,
    "superiori a": Op.GT,
    "superiore ai": Op.GT,
    "superiore al": Op.GT,
    "superiori ai": Op.GT,
    "più di": Op.GT,
    "più alto di": Op.GT,
    "oltre i": Op.GT,
    "oltre": Op.GT,
    "sopra i": Op.GT,
    "sopra a": Op.GT,
    "sopra": Op.GT,
    "minore di": Op.LT,
    "minore a": Op.LT,
    "minori di": Op.LT,
    "inferiore a": Op.LT,
    "inferiori a": Op.LT,
    "inferiore ai": Op.LT,
    "inferiore al": Op.LT,
    "meno di": Op.LT,
    "più basso di": Op.LT,
    "sotto i": Op.LT,
    "sotto a": Op.LT,
    "sotto": Op.LT,
    "almeno": Op.GTE,
    "al massimo": Op.LTE,
    "fino al": Op.LTE,
    "fino alla": Op.LTE,
    "fino a": Op.LTE,
    "entro il": Op.LTE,
    "entro la": Op.LTE,
    "entro": Op.LTE,
    "prima del": Op.LT,
    "prima della": Op.LT,
    "prima dello": Op.LT,
    "prima dell": Op.LT,
    "prima di": Op.LT,
    "dopo il": Op.GT,
    "dopo la": Op.GT,
    "dopo lo": Op.GT,
    "dopo l": Op.GT,
    "dopo del": Op.GT,
    "dopo": Op.GT,
    "successiva al": Op.GT,
    "successivo al": Op.GT,
    "successive al": Op.GT,
    "successivi al": Op.GT,
    "successiva alla": Op.GT,
    "successiva a": Op.GT,
    "precedente al": Op.LT,
    "precedenti al": Op.LT,
    "anteriore al": Op.LT,
    "anteriore a": Op.LT,
    "posteriore al": Op.GT,
    "posteriore a": Op.GT,
    "a partire dal": Op.GTE,
    "a partire dalla": Op.GTE,
    "dal": Op.GTE,
    "dalla": Op.GTE,
    "dall": Op.GTE,
    "pari a": Op.EQ,
    "uguale a": Op.EQ,
}
_OP_WORDS = "|".join(re.escape(k) for k in sorted(_OPERATORS, key=len, reverse=True))
#: Words that sit between a field name and its operator or value.
_FIELD_FILLER = r"(?:is|of|è|e|sia|di|del|della|dello|dei|degli|delle|con|=|:)"
#: Words that follow a field name without being its value.
_FILLER = frozenset(
    {
        "is", "are", "of", "the", "a", "an", "and", "or", "recorded", "values", "value",
        "è", "sono", "di", "del", "della", "dei", "degli", "delle", "il", "lo", "la", "i",
        "gli", "le", "e", "o", "per", "con", "registrati", "registrate", "valori", "valore",
        "distinti", "distinte", "diversi", "diverse",
    }
)  # fmt: skip
_CURRENCY_WORD = r"(?:euro|eur|€|usd|\$)"
_NUMBER_VALUE = rf"(?:[€$]\s*)?(\d[\d.,]*)(?:\s*{_CURRENCY_WORD})?"
_QUOTED = re.compile(r"[\"“«']([^\"”»']{1,80})[\"”»']")

_COUNT = re.compile(
    r"\b(how many|count|number of|quant[ie]|numero (?:di|delle|dei|degli)|conta(?:re)?|"
    r"conteggio)\b",
    re.I,
)
_DISTINCT = re.compile(r"\b(distinct|different|distint[ie]|divers[ie])\b", re.I)
_SUM = re.compile(r"\b(total|sum|totale|somma|complessiv\w+)\b", re.I)
_AVG = re.compile(r"\b(average|mean|media|medio)\b", re.I)
_MAX = re.compile(
    r"\b(highest|largest|maximum|max|most expensive|massim[oa]|più (?:alt|elevat|car)[oa])\b",
    re.I,
)
_MIN = re.compile(
    r"\b(lowest|smallest|minimum|min|cheapest|minim[oa]|più (?:bass|economic)[oa])\b", re.I
)
_GROUP = (
    r"\b(?:grouped by|by|for each|per ogni|per ciascun[oa]?|raggruppat\w+ per|"
    r"suddivis\w+ per|per)\s+(?:(?:il|la|lo|i|gli|le|the)\s+)?"
)
_MONTH_YEAR = rf"(?:(?:il|la|mese di|di)\s+)?({_MONTH_NAMES})\s+(?:del\s+)?(\d{{4}})"
_YEAR = r"(?:(?:nel|in|del|anno)\s+)?((?:19|20)\d{2})(?![\d/.\-])"


def _op_from(token: str | None, default: Op) -> Op:
    return _OPERATORS.get((token or "").strip().lower(), default)


def _normalise(text: str) -> str:
    """Elisions become two words ("dell'anno" -> "dell anno"); quotes stay quotes.

    Only an apostrophe *between* letters is an elision. Replacing every one
    turned "'Bianchi Spa'" into two bare words, and the value lost its second.
    """
    return _ELISION.sub(" ", text)


#: An apostrophe between letters: straight, backtick, or typographic (U+2019).
_ELISION = re.compile(r"(?<=\w)['`" + chr(0x2019) + r"](?=\w)")


def _fields_in(
    text: str, lexicon: Sequence[str], aliases: dict[str, list[str]] | None = None
) -> list[str]:
    """Fields the text names, longest match first and no overlaps.

    "version major" contains "version", so a naive substring scan yields both
    and conjoins a predicate on each -- and a unit carrying `version_major` but
    no `version` then matches neither.
    """
    return [f for f, _ in _field_spans(text, lexicon, aliases or {})]


_WORD = re.compile(r"\w+")


def _stem(word: str) -> str:
    """The form labels and questions are compared in: folded, lightly stemmed.

    So "importi" finds the field ``importo`` and "fatture" the value "fattura":
    Italian inflects nearly every noun, and a lexicon of exact forms misses
    half the questions that name a field. On English words of the lexicon the
    light stemmer changes little, and it changes a label and the question alike.
    """
    return italian_light_stem(fold_accents(word.lower()))


def _stemmed_tokens(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), _stem(m.group(0))) for m in _WORD.finditer(text)]


def _find_label(
    tokens: Sequence[tuple[int, int, str]], label: Sequence[str], taken: list[tuple[int, int]]
) -> tuple[int, int] | None:
    n = len(label)
    for k in range(len(tokens) - n + 1):
        if all(tokens[k + i][2] == label[i] for i in range(n)):
            i, j = tokens[k][0], tokens[k + n - 1][1]
            if not any(i < b and a < j for a, b in taken):
                return i, j
    return None


def _field_spans(
    text: str, lexicon: Sequence[str], aliases: dict[str, list[str]]
) -> list[tuple[str, tuple[int, int, str]]]:
    tokens = _stemmed_tokens(text)
    taken: list[tuple[int, int]] = []
    found: list[tuple[str, tuple[int, int, str]]] = []
    labels: list[tuple[str, str]] = [(f.replace("_", " "), f) for f in lexicon]
    for f, words in aliases.items():
        labels.extend((w, f) for w in words)
    # Longest label first, so "version major" is claimed before "version".
    for label, f in sorted(labels, key=lambda lf: (len(lf[0].split()), len(lf[0])), reverse=True):
        stems = [_stem(w) for w in _WORD.findall(label)]
        if not stems:
            continue
        hit = _find_label(tokens, stems, taken)
        if hit is None:
            continue
        taken.append(hit)
        found.append((f, (hit[0], hit[1], text[hit[0] : hit[1]])))
    found.sort(key=lambda x: x[1][0])
    seen: set[str] = set()
    out: list[tuple[str, tuple[int, int, str]]] = []
    for f, where in found:
        if f not in seen:
            seen.add(f)
            out.append((f, where))
    return out


@dataclass(frozen=True, slots=True)
class RulesRouterParams:
    #: Index names per path. Supplied by the assembler from ``route.paths``.
    paths: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_top_k: int = 50
    #: Field names the structured index knows about. A question naming one is a
    #: strong structured signal, and this is how a corpus teaches the router its
    #: own vocabulary without a code change.
    field_lexicon: list[str] = field(default_factory=list)
    #: Declared type per field (str/int/float/bool/date). Supplied from the
    #: extraction config. Without it the router guesses from the value's
    #: spelling, and a dotted version string reads as a float.
    field_types: dict[str, str] = field(default_factory=dict)
    #: Other words users type for a field: {"importo": ["totale", "euro"]}.
    field_aliases: dict[str, list[str]] = field(default_factory=dict)
    #: Words that mean a field *value*: {"tipo_documento": {"fattura": ["fatture"]}}.
    value_aliases: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    #: Where an unattached date constraint applies ("nel 2024", "dopo il 30/06").
    default_date_field: str = ""
    #: Where an unattached amount applies ("sopra i 1000 euro"), and what SUM,
    #: AVG, MAX and MIN aggregate when the question names no numeric field.
    default_measure: str = ""
    #: How numbers in questions are written: it, en or auto.
    locale: str = "auto"
    #: unit | document. What a structured row is, and so what a count counts.
    level: str = "unit"
    limit: int = 50
    iterative_min_words: int = 14
    enable_structured: bool = True


@register(
    "route",
    "rules",
    version="2",
    params_model=dataclass_params(RulesRouterParams),
    summary=(
        "Regex and field-lexicon rules, English and Italian. Zero latency, "
        "deterministic, a strong baseline."
    ),
)
def _make_rules(params: dict[str, Any], **_: Any) -> RulesRouter:
    return RulesRouter(params)


class RulesRouter(StageImpl):
    STAGE, IMPL, VERSION = "route", "rules", "2"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.paths: dict[str, dict[str, Any]] = params.get("paths", {})
        self.lexicon = [w.lower() for w in params.get("field_lexicon", [])]
        self.field_types: dict[str, str] = {
            k: str(v).lower() for k, v in (params.get("field_types") or {}).items()
        }
        self.aliases: dict[str, list[str]] = {
            k: [str(w) for w in v] for k, v in (params.get("field_aliases") or {}).items()
        }
        self.value_aliases: dict[str, dict[str, list[str]]] = params.get("value_aliases") or {}
        self.default_date_field = str(params.get("default_date_field") or "")
        self.default_measure = str(params.get("default_measure") or "")
        self.locale = str(params.get("locale", "auto"))
        self.level = str(params.get("level", "unit"))
        self.limit = int(params.get("limit", 50))
        self.default_top_k = int(params.get("default_top_k", 50))
        self.iterative_min_words = int(params.get("iterative_min_words", 14))
        self.enable_structured = bool(params.get("enable_structured", True))

    def route(self, query: Query, ctx: StageContext) -> RouteDecision:
        text = query.text
        qtype, reason = self._classify(text)

        wants_structured = qtype in (
            QueryType.STRUCTURED,
            QueryType.NUMERIC,
            QueryType.TEMPORAL,
        )
        if wants_structured and self.enable_structured:
            sq = self._structured_query(text, qtype)
            if sq is not None and self._targets("structured"):
                return RouteDecision(
                    path=RoutePath.STRUCTURED,
                    targets=self._targets("structured"),
                    step_budget=1,
                    structured_query=sq,
                    query_type=qtype,
                    confidence=0.8,
                    reason=reason,
                    router=self.IMPL,
                    fingerprint=self.fingerprint().key(),
                )
            # Nothing extractable from the text. Fall back to lookup, but
            # record *why* -- see below.
            reason = f"{reason}; no structured predicate extractable, falling back to lookup"
        elif wants_structured:
            # The question wanted the structured path and the configuration does
            # not offer one. An operator reading the decision log needs to see
            # that, not a line saying the question looked like prose: the fix is
            # in the config, and nothing else in the trace would point there.
            reason = f"{reason}; structured path unavailable (no structured index), using lookup"

        if qtype in (QueryType.MULTI_HOP, QueryType.COMPARATIVE) or (
            len(text.split()) >= self.iterative_min_words and qtype is QueryType.SUMMARY
        ):
            spec = self.paths.get("iterative", {})
            return RouteDecision(
                path=RoutePath.ITERATIVE,
                targets=self._targets("iterative"),
                step_budget=int(spec.get("step_budget", 3)),
                query_type=qtype,
                sub_queries=self._decompose(text),
                confidence=0.6,
                reason=reason,
                router=self.IMPL,
                fingerprint=self.fingerprint().key(),
            )

        return RouteDecision(
            path=RoutePath.LOOKUP,
            targets=self._targets("lookup"),
            step_budget=1,
            query_type=qtype,
            confidence=0.7,
            reason=reason,
            router=self.IMPL,
            fingerprint=self.fingerprint().key(),
        )

    # ------------------------------------------------------------- internals

    def _mentions(self, text: str) -> list[str]:
        """Fields named directly, by alias, or through one of their values."""
        norm = _normalise(text)
        named = _fields_in(norm, self.lexicon, self.aliases)
        for f in self._value_matches(norm):
            if f[0] not in named:
                named.append(f[0])
        return named

    def _classify(self, text: str) -> tuple[QueryType, str]:
        norm = _normalise(text)
        mentions_field = self._mentions(norm)
        has_default_date = bool(self.default_date_field) and bool(_DATE.search(norm))
        if _AGGREGATE.search(norm) and (mentions_field or _NUMERIC.search(norm)):
            return (
                QueryType.STRUCTURED,
                f"aggregate + field/numeric ({', '.join(mentions_field) or 'numeric'})",
            )
        if _NUMERIC.search(norm):
            return QueryType.NUMERIC, "numeric comparison"
        if _TEMPORAL.search(norm) and (mentions_field or _DATE.search(norm)):
            return QueryType.TEMPORAL, "temporal constraint"
        if _COMPARATIVE.search(norm):
            return QueryType.COMPARATIVE, "comparative"
        if _MULTI_HOP.search(norm):
            return QueryType.MULTI_HOP, "multi-hop phrasing"
        if _SUMMARY.search(norm):
            return QueryType.SUMMARY, "summary request"
        if mentions_field and _AGGREGATE.search(norm):
            return QueryType.STRUCTURED, f"field mention: {mentions_field[0]}"
        if has_default_date and self._value_matches(norm):
            return QueryType.TEMPORAL, "document type + date"
        return QueryType.FACTUAL, "no structural signal; factual lookup"

    def _structured_query(self, text: str, qtype: QueryType | None) -> StructuredQuery | None:
        """Extract a predicate from the text. Conservative by design.

        Returning ``None`` when nothing is confidently extractable is the right
        behaviour: a STRUCTURED route with a wrong predicate returns a confident
        wrong answer, which is worse than falling back to retrieval.

        Four things here are easy to get wrong, and all four were, until the
        structured slice of a real golden set showed well-formed questions
        returning nothing:

        *Dates are not numbers.* "release date before 2023-06-27" read as
        ``release_date == 2023`` compares a date column against an integer and
        matches nothing. Dates are parsed before numbers.

        *Temporal words are operators.* before, after, since, until carry the
        comparison; without them every temporal question collapsed to equality.

        *"greater than" is an operator too.* The lexicon knew ``>`` and "over"
        but not the words most people actually type.

        *The longest field name wins.* "version major" contains "version", so a
        substring match produced a predicate on both fields conjoined -- and a
        unit with a ``version_major`` but no ``version`` matched neither.
        """
        norm = _normalise(text)
        spans = _field_spans(norm, self.lexicon, self.aliases)
        fields_named = [f for f, _ in spans]
        value_hits = self._value_matches(norm)
        if not fields_named and not value_hits and not self._defaults_apply(norm):
            return None

        group_by = self._group_by(norm, spans)
        clauses: list[Predicate] = []
        constrained: set[str] = set()
        # Text a clause has consumed -- a field, its operator and its value --
        # is blanked before looking for constraints that name no field, so the
        # "31" of "entro il 31/12/2025" is not read again as an amount.
        rest = norm
        for f, (_, _, label) in spans:
            found = self._clause_for(norm, f, label, grouping=f in group_by)
            if found is not None:
                clause, (i, j) = found
                clauses.append(clause)
                constrained.add(f)
                rest = rest[:i] + " " * (j - i) + rest[j:]
        for f, value in value_hits:
            if f not in constrained:
                clauses.append(Compare(f, Op.EQ, value))
                constrained.add(f)
        clauses.extend(self._unattached(rest, constrained))

        masked = norm
        for _, (i, j, _) in sorted(spans, key=lambda s: -s[1][0]):
            masked = masked[:i] + " " * (j - i) + masked[j:]
        aggregations = self._aggregations(masked, fields_named)
        if not clauses and not aggregations:
            return None
        select = tuple(group_by) if group_by else ()
        if not aggregations and not group_by:
            select = tuple(f for f in fields_named if f not in group_by)[:4] or ("unit_id",)
            if self.level == "document":
                # A document-level row names its document, so whoever reads the
                # answer -- an agent, usually -- can open what it is about.
                select = ("document_id", *(f for f in select if f != "unit_id"))
        return StructuredQuery(
            where=all_of(clauses),
            select=select,
            group_by=tuple(group_by),
            aggregations=aggregations,
            limit=self.limit,
            level=self.level,
        )

    # ---------------------------------------------------------- predicates

    def _clause_for(
        self, text: str, f: str, label: str, *, grouping: bool
    ) -> tuple[Predicate, tuple[int, int]] | None:
        """The constraint on one named field, and the span of text it used."""
        window = rf"(?<!\w){re.escape(label)}\s*(?:{_FIELD_FILLER}\s+)?(?:({_OP_WORDS})\s*)?"
        declared = self.field_types.get(f)

        if declared in (None, "date", "datetime"):
            found = self._date_clause(text, window, f, allow_year=declared is not None)
            if found is not None:
                return found

        # A field declared textual takes a textual value, however numeric the
        # value looks. "version 1.0.0" is a string in every corpus that has ever
        # had a patch release, and reading it as 1.0 queries a numeric column
        # the value was never written to.
        if declared in (None, "int", "float", "bool"):
            m = re.search(window + _NUMBER_VALUE + r"(?![\w.\-])", text, re.I)
            if m:
                value = self._number(m.group(2), declared)
                if value is not None:
                    return Compare(f, _op_from(m.group(1), default=Op.EQ), value), m.span()

        m = re.search(window + r"(?:" + _QUOTED.pattern + r")", text, re.I)
        if m:
            return self._text_clause(f, m.group(1), m.group(2)), m.span()
        m = re.search(window + r"([^\W_][\w.\-/]{0,40})", text, re.I)
        if m and m.group(2).lower() not in _FILLER and not grouping:
            return self._text_clause(f, m.group(1), m.group(2)), m.span()
        if grouping:
            return None
        m = re.search(rf"(?<!\w){re.escape(label)}", text, re.I)
        return Exists(f), (m.span() if m else (0, 0))

    def _text_clause(self, f: str, op_word: str | None, value: str) -> Predicate:
        op = _op_from(op_word, default=Op.EQ)
        return Compare(f, op, value) if op is not Op.EQ else TextMatch(f, value, mode="exact")

    def _number(self, raw: str, declared: str | None) -> int | float | None:
        value = parse_number(raw, self.locale)
        if value is None:
            return None
        if declared == "float":
            return value
        if declared == "int" or value == int(value):
            return int(value)
        return value

    def _date_clause(
        self, text: str, window: str, f: str, *, allow_year: bool
    ) -> tuple[Predicate, tuple[int, int]] | None:
        """A date constraint after ``window``: a day, a month, or (for declared
        date fields) a year -- the last two as ranges. With the span it used."""
        for pattern in (
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})(?![\d/.\-])",
            rf"(\d{{1,2}}(?:°|º)?\s+(?:{_MONTH_NAMES})\s+\d{{4}})",
        ):
            m = re.search(window + pattern, text, re.I)
            if m:
                d = parse_date(m.group(2))
                if d is not None:
                    return Compare(f, _op_from(m.group(1), default=Op.EQ), d), m.span()
        m = re.search(window + _MONTH_YEAR, text, re.I)
        if m:
            month = _month_number(m.group(2))
            if month:
                first = date(int(m.group(3)), month, 1)
                last = date(first.year, month, calendar.monthrange(first.year, month)[1])
                return _range(f, _op_from(m.group(1), default=Op.EQ), first, last), m.span()
        if allow_year:
            m = re.search(window + _YEAR, text, re.I)
            if m:
                year = int(m.group(2))
                period = _range(
                    f, _op_from(m.group(1), default=Op.EQ), date(year, 1, 1), date(year, 12, 31)
                )
                return period, m.span()
        return None

    def _defaults_apply(self, text: str) -> bool:
        return bool(
            (self.default_date_field and _DATE.search(text))
            or (self.default_measure and re.search(_NUMBER_VALUE, text))
        )

    def _unattached(self, text: str, constrained: set[str]) -> list[Predicate]:
        """Constraints the question states without naming a field.

        "fatture emesse nel 2024" names no date field; with a configured
        ``default_date_field`` the year still applies to it. Likewise an amount
        with an operator or a currency ("sopra i 1000 euro") and
        ``default_measure``. ``text`` has the spans earlier clauses consumed
        blanked out, and each constraint found here blanks its own.
        """
        out: list[Predicate] = []
        anchor = rf"(?<!\w)(?:({_OP_WORDS})\s+)?"
        if self.default_date_field and self.default_date_field not in constrained:
            f = self.default_date_field
            found = self._date_clause(text, anchor, f, allow_year=False)
            if found is None:
                m = re.search(
                    rf"(?:(?<!\w)({_OP_WORDS})\s+|\b(?:nel|in|del|during|durante|anno)\s+)"
                    r"((?:19|20)\d{2})(?![\d/.\-])",
                    text,
                    re.I,
                )
                if m:
                    year = int(m.group(2))
                    op = _op_from(m.group(1), default=Op.EQ)
                    found = _range(f, op, date(year, 1, 1), date(year, 12, 31)), m.span()
            if found is not None:
                clause, (i, j) = found
                out.append(clause)
                text = text[:i] + " " * (j - i) + text[j:]
        if self.default_measure and self.default_measure not in constrained:
            # An amount needs an operator or a currency to be one: a bare number
            # in a question is as likely a quantity, an id or a page.
            m = re.search(
                rf"(?<!\w)({_OP_WORDS})\s+(?:[€$]\s*)?(\d[\d.,]*)(?![\w/])"
                rf"|[€$]\s*(\d[\d.,]*)|(\d[\d.,]*)\s*{_CURRENCY_WORD}(?!\w)",
                text,
                re.I,
            )
            if m:
                raw = m.group(2) or m.group(3) or m.group(4)
                value = self._number(raw, self.field_types.get(self.default_measure))
                if value is not None:
                    out.append(Compare(self.default_measure, _op_from(m.group(1), Op.EQ), value))
        return out

    def _value_matches(self, text: str) -> list[tuple[str, Any]]:
        """(field, value) for every configured value synonym the text contains,
        compared in stemmed form, so "fatture" finds the value "fattura"."""
        tokens = _stemmed_tokens(text)
        out: list[tuple[str, Any]] = []
        for f, values in self.value_aliases.items():
            for value, words in values.items():
                for w in [str(value), *words]:
                    stems = [_stem(x) for x in _WORD.findall(w)]
                    if stems and _find_label(tokens, stems, []) is not None:
                        out.append((f, value))
                        break
                else:
                    continue
                break
        return out

    # ---------------------------------------------------------- aggregates

    def _group_by(self, text: str, spans: Sequence[tuple[str, tuple[int, int, str]]]) -> list[str]:
        out: list[str] = []
        for f, (_, _, label) in spans:
            if re.search(_GROUP + re.escape(label) + r"(?!\w)", text, re.I):
                out.append(f)
        return out

    def _aggregations(self, masked: str, fields_named: Sequence[str]) -> tuple[Aggregation, ...]:
        """The aggregate the question asks for, if any -- over the field labels
        masked out, so "importo totale" is a field and not a request to sum."""
        numeric = [f for f in fields_named if self.field_types.get(f) in ("int", "float")] or (
            [self.default_measure] if self.default_measure else []
        )
        if _COUNT.search(masked):
            if _DISTINCT.search(masked) and fields_named:
                return (Aggregation(AggregationOp.COUNT, fields_named[0], distinct=True),)
            return (Aggregation(AggregationOp.COUNT),)
        for rx, op in ((_SUM, AggregationOp.SUM), (_AVG, AggregationOp.AVG),
                       (_MAX, AggregationOp.MAX), (_MIN, AggregationOp.MIN)):  # fmt: skip
            if rx.search(masked) and numeric:
                return (Aggregation(op, numeric[0]),)
        return ()

    def _decompose(self, text: str) -> tuple[str, ...]:
        """Split a comparative or multi-hop question into retrievable parts."""
        parts = re.split(
            r"\b(?:versus|vs\.?|compared to|and then|as well as|rispetto a|e poi|oltre a)\b",
            text,
            flags=re.I,
        )
        cleaned = [p.strip(" ,?.") for p in parts if len(p.strip()) > 8]
        return tuple(cleaned) if len(cleaned) > 1 else (text,)

    def _targets(self, path: str) -> tuple[RouteTarget, ...]:
        spec = self.paths.get(path, {})
        top_k = spec.get("top_k", {})
        return tuple(
            RouteTarget(index=name, top_k=int(top_k.get(name, self.default_top_k)))
            for name in spec.get("targets", [])
        )


def _month_number(name: str) -> int | None:
    d = parse_date(f"1 {name} 2000")
    return d.month if d else None


def _range(f: str, op: Op, first: date, last: date) -> Predicate:
    """A period as a constraint: equality means "within", comparisons use the
    edge that makes them exclusive of the period itself."""
    match op:
        case Op.GT:
            return Compare(f, Op.GT, last)
        case Op.GTE:
            return Compare(f, Op.GTE, first)
        case Op.LT:
            return Compare(f, Op.LT, first)
        case Op.LTE:
            return Compare(f, Op.LTE, last)
    return And((Compare(f, Op.GTE, first), Compare(f, Op.LTE, last)))


# --------------------------------------------------------------------------- #
# passthrough                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PassthroughRouterParams:
    targets: list[str] = field(default_factory=list)
    top_k: int = 50


@register(
    "route",
    "passthrough",
    version="1",
    params_model=dataclass_params(PassthroughRouterParams),
    summary="Everything takes LOOKUP against all indexes. The no-router ablation arm.",
)
def _make_passthrough_router(params: dict[str, Any], **_: Any) -> PassthroughRouter:
    return PassthroughRouter(params)


class PassthroughRouter(StageImpl):
    """One path for everything.

    Deliberately violates invariant 5, which is the point: the gap between this
    and the rules router *is* the measured value of routing, and without an arm
    that does the wrong thing there is no number.
    """

    STAGE, IMPL, VERSION = "route", "passthrough", "1"

    def route(self, query: Query, ctx: StageContext) -> RouteDecision:
        return RouteDecision(
            path=RoutePath.LOOKUP,
            targets=tuple(
                RouteTarget(index=n, top_k=int(self.param("top_k", 50)))
                for n in self.param("targets", [])
            ),
            step_budget=1,
            query_type=QueryType.UNKNOWN,
            confidence=0.0,
            reason="passthrough: no routing performed",
            router=self.IMPL,
            fingerprint=self.fingerprint().key(),
        )

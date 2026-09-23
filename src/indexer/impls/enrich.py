"""Enrichers: section prefix, extractive context, regex fields, LLM contextualiser.

Invariant 3 is won here. The four implementations form a ladder that separates
three questions the single number "contextualisation helps" conflates:

``section_prefix``     Does *any* prefix help? (the heading trail, no model)
``extractive_context`` Does a *document-aware* prefix help? (still no model)
``llm_contextualizer`` Does an *LLM-written summary* help? (a model)

Without the first two, a measured gain from the third cannot be attributed: a
prefix that merely repeats the document title would show most of the same
benefit, and concluding "the LLM summary is worth it" from that would be wrong.

``regex_fields`` is the fourth, and serves invariant 5 rather than 3: the
structured path can only answer from fields that were extracted.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from indexer.core.ids import hash_obj
from indexer.core.registry import register
from indexer.core.stages import EnrichContext
from indexer.core.unit import ContextScope, Enrichment, FieldValue, Unit
from indexer.plugin import StageImpl, dataclass_params
from indexer.textutil import STOPWORDS, WORD_RE, fold

__all__ = [
    "ExtractiveContextualizer",
    "LLMContextualizer",
    "RegexFieldExtractor",
    "SectionPrefixEnricher",
]


# --------------------------------------------------------------------------- #
# section prefix -- the no-model control                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SectionPrefixParams:
    include_document_title: bool = True
    separator: str = " > "


@register(
    "enrich",
    "section_prefix",
    version="1",
    params_model=dataclass_params(SectionPrefixParams),
    summary="Prepends the heading trail. No model. The control arm for contextualisation.",
)
def _make_section_prefix(params: dict[str, Any], **_: Any) -> SectionPrefixEnricher:
    return SectionPrefixEnricher(params)


class SectionPrefixEnricher(StageImpl):
    """Prepends the document title and heading trail.

    Unit-scoped: it reads only the unit's own ``section_path``, so an edit
    elsewhere in the document does not invalidate it. That is what unit scope is
    for, and this is the clearest example of it.
    """

    STAGE, IMPL, VERSION = "enrich", "section_prefix", "1"
    name = "section_prefix"
    scope = ContextScope.UNIT

    def enrich(self, units: Sequence[Unit], ctx: EnrichContext) -> Sequence[Enrichment]:
        sep = self.param("separator", " > ")
        include_title = self.param("include_document_title", True)
        title = _document_title(ctx) if include_title else ""
        fp = self.fingerprint().key()
        out = []
        for u in units:
            trail = [t for t in ((title,) if title else ()) + u.section_path if t]
            out.append(
                Enrichment(
                    enricher=self.name,
                    fingerprint=fp,
                    context=sep.join(trail) if trail else None,
                    scope=self.scope,
                )
            )
        return out


def _document_title(ctx: EnrichContext) -> str:
    for b in ctx.document.blocks:
        if str(b.kind) == "heading":
            return b.text
    name = str(ctx.document.metadata.get("name", ""))
    return name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ") if name else ""


# --------------------------------------------------------------------------- #
# extractive context -- document-aware, still no model                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ExtractiveParams:
    max_chars: int = 400
    #: Sentences pulled from the document's opening, which is where a README or
    #: a chapter states what it is about.
    lead_sentences: int = 2
    include_section_path: bool = True
    include_neighbors: bool = False
    #: Distinctive terms from the parent document that the unit itself omits.
    #: This is the part that does the work an LLM summary would: it supplies the
    #: subject a pronoun-heavy chunk never names.
    distinctive_terms: int = 6


@register(
    "enrich",
    "extractive_context",
    version="2",
    params_model=dataclass_params(ExtractiveParams),
    summary=(
        "Document lead + heading trail + distinctive parent terms the unit omits. "
        "No model; the offline stand-in for LLM contextualisation."
    ),
)
def _make_extractive(params: dict[str, Any], **_: Any) -> ExtractiveContextualizer:
    return ExtractiveContextualizer(params)


class ExtractiveContextualizer(StageImpl):
    """Document-aware context without a model.

    What it is approximating: an LLM-written 50-100 token summary that situates
    the chunk in its document. What it actually does: takes the document's lead
    sentences, the heading trail, and the terms the *document* is about that
    this *unit* never names.

    That last part targets the specific failure contextualisation fixes. A chunk
    reading "It must be set before the first request" is unretrievable for
    "requests timeout" because neither word appears in it. Supplying the
    document's distinctive terms puts them in the retrieval surface for both the
    dense and lexical halves.

    Document-scoped, necessarily: it reads the parent. Any edit to the document
    invalidates every unit's context, which is the honest cost of document-level
    context and exactly what ``ContextScope`` exists to make visible.
    """

    STAGE, IMPL, VERSION = "enrich", "extractive_context", "2"
    name = "extractive_context"
    scope = ContextScope.DOCUMENT

    def enrich(self, units: Sequence[Unit], ctx: EnrichContext) -> Sequence[Enrichment]:
        max_chars = int(self.param("max_chars", 400))
        n_lead = int(self.param("lead_sentences", 2))
        n_terms = int(self.param("distinctive_terms", 6))
        fp = self.fingerprint().key()

        title = _document_title(ctx)
        lead = _lead_sentences(ctx.document.text, n_lead)
        doc_terms = _top_terms(ctx.document.text, n_terms * 3)

        out: list[Enrichment] = []
        for u in units:
            parts: list[str] = []
            if title:
                parts.append(title)
            if self.param("include_section_path", True) and u.section_path:
                parts.append(" > ".join(u.section_path))
            if lead:
                parts.append(lead)
            body_terms = _tokenize(u.text)
            missing = [t for t in doc_terms if t not in body_terms][:n_terms]
            if missing:
                parts.append("Topics: " + ", ".join(missing) + ".")
            context = " ".join(parts)[:max_chars].strip()
            out.append(
                Enrichment(
                    enricher=self.name,
                    fingerprint=fp,
                    context=context or None,
                    scope=self.scope,
                )
            )
        return out


def _tokenize(text: str) -> set[str]:
    return {m.group(0) for m in WORD_RE.finditer(fold(text))}


def _top_terms(text: str, n: int) -> list[str]:
    """Frequent, non-stopword terms. A crude TF proxy for what a document is about."""
    counts: dict[str, int] = {}
    for m in WORD_RE.finditer(fold(text)):
        w = m.group(0)
        if w in STOPWORDS or w.isdigit():
            continue
        counts[w] = counts.get(w, 0) + 1
    # Ties broken alphabetically: the context string is content-hashed, and a
    # dict-order-dependent tiebreak would produce cache misses that look like
    # corruption.
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _lead_sentences(text: str, n: int) -> str:
    body = text.strip()
    if not body:
        return ""
    # Skip a title line: it is already supplied separately, and repeating it
    # spends the context budget on a term that is already present.
    lines = [ln for ln in body.splitlines() if ln.strip()]
    prose = " ".join(lines[1:]) if len(lines) > 1 else body
    return " ".join(_SENT_SPLIT.split(prose)[:n]).strip()


# --------------------------------------------------------------------------- #
# regex fields -- invariant 5's supply side                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RegexFieldParams:
    #: field name -> {pattern, type, group}. Kept in config so a new corpus adds
    #: fields without touching code, which is the whole claim of the frame.
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Copy scanner metadata straight through as fields (package name, version).
    from_metadata: list[str] = field(default_factory=list)


@register(
    "enrich",
    "regex_fields",
    version="1",
    params_model=dataclass_params(RegexFieldParams),
    summary="Typed field extraction by regex. No model. Feeds the structured index.",
)
def _make_regex_fields(params: dict[str, Any], **_: Any) -> RegexFieldExtractor:
    return RegexFieldExtractor(params)


class RegexFieldExtractor(StageImpl):
    """Typed extraction without a model.

    Real corpora carry a surprising number of machine-readable facts -- version
    strings, dates, requirement specifiers -- and a regex gets them exactly
    right where an LLM gets them nearly right. "Nearly right" on a number that
    a structured query will compare is worse than useless.
    """

    STAGE, IMPL, VERSION = "enrich", "regex_fields", "1"
    name = "regex_fields"
    scope = ContextScope.UNIT

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._compiled = {
            name: (
                re.compile(spec["pattern"], re.I | re.M),
                spec.get("type", "str"),
                spec.get("group", 1),
            )
            for name, spec in params.get("fields", {}).items()
        }
        self._from_metadata: list[str] = params.get("from_metadata", [])

    def enrich(self, units: Sequence[Unit], ctx: EnrichContext) -> Sequence[Enrichment]:
        fp = self.fingerprint().key()
        out: list[Enrichment] = []
        for u in units:
            fields_out: dict[str, FieldValue] = {}
            for key in self._from_metadata:
                if key in u.metadata:
                    fields_out[key] = u.metadata[key]
            for name, (rx, typ, group) in self._compiled.items():
                m = rx.search(u.text)
                if m:
                    value = _coerce(m.group(group) if m.lastindex else m.group(0), typ)
                    if value is not None:
                        fields_out[name] = value
            out.append(
                Enrichment(
                    enricher=self.name,
                    fingerprint=fp,
                    fields=fields_out,
                    scope=self.scope,
                )
            )
        return out


def _coerce(raw: str, typ: str) -> FieldValue:
    raw = raw.strip()
    try:
        match typ:
            case "int":
                return int(re.sub(r"[,_\s]", "", raw))
            case "float":
                return float(re.sub(r"[,_\s]", "", raw))
            case "bool":
                return raw.lower() in ("true", "yes", "1")
            case "date":
                return date.fromisoformat(raw[:10])
            case _:
                return raw
    except (ValueError, TypeError):
        # A value that does not parse is dropped rather than stored as a string.
        # Storing "circa 2019" in a date column makes every temporal predicate
        # over that column raise or silently mis-sort.
        return None


# --------------------------------------------------------------------------- #
# LLM contextualiser                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LLMContextParams:
    model: str = "claude-haiku-4-5-20251001"
    target_tokens: int = 75
    max_tokens: int = 128
    prompt_cache: bool = True
    system: str = ""
    max_document_chars: int = 200_000


@register(
    "enrich",
    "llm_contextualizer",
    version="1",
    params_model=dataclass_params(LLMContextParams),
    summary=(
        "LLM-written 50-100 token situating summary, prompt-cached over the parent "
        "document. Requires ANTHROPIC_API_KEY."
    ),
    requires=("anthropic",),
)
def _make_llm_context(params: dict[str, Any], **kw: Any) -> LLMContextualizer:
    return LLMContextualizer(params, client=kw.get("client"))


class LLMContextualizer(StageImpl):
    """Invariant 3's reference implementation.

    The batching and caching shape is the point. Units arrive batched *by
    document*, and the parent document is sent as a cached prompt prefix, so a
    200-page document is paid for once rather than once per chunk. The brief's
    guidance -- "a small fast model with prompt caching over the parent
    document" -- is a cost structure, not just a model choice, and the
    ``EnrichContext`` shape exists to make it expressible.

    ``VERSION`` must be bumped whenever the prompt changes. An edited prompt
    with the same version serves stale summaries forever, and the symptom is the
    edit appearing to do nothing.
    """

    STAGE, IMPL, VERSION = "enrich", "llm_contextualizer", "1"
    name = "llm_contextualizer"
    scope = ContextScope.DOCUMENT

    PROMPT = (
        "Here is a chunk from the document above:\n<chunk>\n{chunk}\n</chunk>\n\n"
        "Write a short standalone summary ({target} tokens) that situates this chunk "
        "within the document, so it can be found by search. Name the subject "
        "explicitly rather than using pronouns. Answer with the summary only."
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
                    "llm_contextualizer needs the `anthropic` package: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def enrich(self, units: Sequence[Unit], ctx: EnrichContext) -> Sequence[Enrichment]:
        client = self._get_client()
        model = self.param("model")
        target = int(self.param("target_tokens", 75))
        fp = self.fingerprint().key()
        doc_text = ctx.document.text[: int(self.param("max_document_chars", 200_000))]

        # The document block is marked for caching, so it is billed once for the
        # batch rather than once per unit. This is what makes per-chunk
        # contextualisation affordable at corpus scale.
        doc_block: dict[str, Any] = {"type": "text", "text": f"<document>\n{doc_text}\n</document>"}
        if self.param("prompt_cache", True):
            doc_block["cache_control"] = {"type": "ephemeral"}

        out: list[Enrichment] = []
        for u in units:
            resp = client.messages.create(
                model=model,
                max_tokens=int(self.param("max_tokens", 128)),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            doc_block,
                            {
                                "type": "text",
                                "text": self.PROMPT.format(chunk=u.text, target=target),
                            },
                        ],
                    }
                ],
            )
            usage = getattr(resp, "usage", None)
            out.append(
                Enrichment(
                    enricher=self.name,
                    fingerprint=fp,
                    context=resp.content[0].text.strip(),
                    scope=self.scope,
                    tokens_in=getattr(usage, "input_tokens", 0) if usage else 0,
                    tokens_out=getattr(usage, "output_tokens", 0) if usage else 0,
                )
            )
        return out

    def fingerprint(self) -> Any:
        from indexer.core.accounting import StageFingerprint

        # The prompt is part of the fingerprint, so editing the template
        # invalidates the cache even if VERSION is forgotten. Belt and braces,
        # because this is the failure mode that costs the most to diagnose.
        return StageFingerprint(
            stage=self.STAGE,
            impl=self.IMPL,
            version=self.VERSION,
            params_hash=hash_obj({"params": self._params, "prompt": self.PROMPT}),
        )

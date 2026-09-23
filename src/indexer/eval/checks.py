"""Contract conformance checks.

A contract stated only in a docstring is a suggestion. These functions make the
stage contracts executable, so a new implementation is tested against the same
rules as the reference one, and a violation is a failed check with a message
rather than a mysterious quality regression three stages downstream.

Every check takes real stage output and returns a list of violations. They are
used three ways: in the conformance test suite that every implementation is
expected to pass, as optional runtime assertions during a build
(``--strict-contracts``), and by the eval harness before a run -- because a
golden set scored against a corpus whose provenance is broken produces numbers
that look fine and mean nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise

from indexer.core.document import BlockKind, ParsedDocument
from indexer.core.results import RankedList
from indexer.core.stages import Index, IndexQuery, StageContext
from indexer.core.unit import EnrichedUnit, Unit

__all__ = [
    "check_context_specificity",
    "check_index_surface",
    "check_parsed_document",
    "check_ranked_list",
    "check_unit_stability",
    "check_units",
]


def check_parsed_document(parsed: ParsedDocument) -> list[str]:
    """The parse contract: span integrity, reading order, table structure."""
    problems: list[str] = []
    prev_end = 0
    for i, b in enumerate(parsed.blocks):
        span = b.provenance.span
        actual = parsed.text[span.start : span.end]
        if actual != b.text:
            problems.append(
                f"block[{i}] {b.block_id}: span {span.start}..{span.end} yields "
                f"{actual[:40]!r} but block.text is {b.text[:40]!r}. Every citation "
                f"this document ever produces descends from this equality."
            )
        if span.start < prev_end:
            problems.append(
                f"block[{i}] {b.block_id}: span starts at {span.start}, before the "
                f"previous block ended at {prev_end}. Blocks must be ascending and "
                f"non-overlapping -- reading order is the list order."
            )
        prev_end = max(prev_end, span.end)
        if b.provenance.document_id != parsed.document_id:
            problems.append(f"block[{i}]: provenance names a different document")
        if str(b.kind) == BlockKind.TABLE and b.table is None:
            problems.append(
                f"block[{i}] {b.block_id}: kind is 'table' but no Table payload. "
                f"Rendering a table to a string and dropping the grid destroys the "
                f"structured path at the first stage; no later stage recovers it."
            )
        if str(b.kind) == BlockKind.HEADING and b.level is None:
            problems.append(f"block[{i}]: heading without a level; section paths need it")
    if not 0.0 <= parsed.reading_order_confidence <= 1.0:
        problems.append("reading_order_confidence must be in [0, 1]")
    return problems


def check_units(units: Sequence[Unit], parsed: ParsedDocument) -> list[str]:
    """The segment contract: addressability, ordering, coverage."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, u in enumerate(units):
        if u.unit_id in seen:
            problems.append(f"unit[{i}]: duplicate unit_id {u.unit_id}")
        seen.add(u.unit_id)
        if u.document_id != parsed.document_id:
            problems.append(f"unit[{i}]: belongs to a different document")
        span = u.provenance.span
        if span.end > len(parsed.text):
            problems.append(
                f"unit[{i}]: span {span.start}..{span.end} runs past the document "
                f"({len(parsed.text)} chars)"
            )
            continue
        slice_ = parsed.text[span.start : span.end]
        if u.verbatim:
            if u.text and u.text not in slice_:
                problems.append(
                    f"unit[{i}]: text is not the document's text at its own span -- "
                    f"a citation from this unit would point at the wrong place. If the "
                    f"text is a deliberate derivation (a split table with repeated "
                    f"headers), set Unit.verbatim=False so it is declared rather than "
                    f"silently tolerated."
                )
        elif u.text and not _shares_substance(u.text, slice_):
            # A derived unit still has to come from its span. Without this,
            # `verbatim=False` would be a licence to attach any text to any
            # location, which is exactly the failure the flag exists to bound.
            problems.append(
                f"unit[{i}]: declared non-verbatim, but its text has little in common "
                f"with the span it claims to derive from"
            )
    ordinals = [u.ordinal for u in units]
    if ordinals != sorted(ordinals):
        problems.append("units are not in reading order by ordinal")
    return problems


def _shares_substance(text: str, source: str, threshold: float = 0.6) -> bool:
    """Whether a derived unit plausibly came from its span.

    Word-level containment: a split table chunk repeats its header and carries a
    subset of the rows, so nearly all of its words come from the span. A unit
    attached to the wrong span would not.
    """
    words = [w for w in text.split() if len(w) > 2]
    if not words:
        return True
    pool = set(source.split())
    return sum(1 for w in words if w in pool) / len(words) >= threshold


def check_unit_stability(before: Sequence[Unit], after: Sequence[Unit]) -> list[str]:
    """The incremental contract: unchanged content keeps its id.

    Run by editing one paragraph of a document and re-segmenting. Units whose
    text did not change must keep their ids, or an incremental rebuild rewrites
    the whole document and "adding 10 documents to 500 reprocesses 10" is false
    for every document that is ever edited.
    """
    problems: list[str] = []
    by_text_before: dict[str, list[str]] = {}
    for u in before:
        by_text_before.setdefault(u.text, []).append(u.unit_id)
    for u in after:
        prior = by_text_before.get(u.text)
        if prior and u.unit_id not in prior:
            problems.append(
                f"unit with unchanged text changed id ({prior[0]} -> {u.unit_id}); "
                f"position is leaking into unit identity"
            )
    return problems


def check_index_surface(
    index: Index, units: Sequence[EnrichedUnit], ctx: StageContext
) -> list[str]:
    """The index contract: every index searches ``indexing_text()``.

    This is the check that catches invariant 3's most likely failure -- the
    lexical half indexing raw text while the dense half gets the contextualised
    string. It is nearly invisible otherwise: retrieval still works, just worse,
    and the ablation shows contextualisation earning about half of what it
    should.

    Method: pick a unit whose prepended context contains a term absent from its
    own text **and rare across the corpus**, search for that term, and require
    the unit to come back.

    Rarity is not optional. On a real corpus the first context-only term is
    usually the document title or a section name shared by every unit of that
    document -- on the PyPI-docs corpus it was "Changelog", present in 2,181 of
    10,283 retrieval surfaces -- and no correct index can be expected to return
    one particular unit in its top 50 for a term that common. The first run of
    this check on that corpus reported both indexes as broken when both were
    fine. So the probe is the context-only term with the lowest document
    frequency over the retrieval surface, and the search depth is at least that
    frequency.
    """
    problems: list[str] = []

    def words(text: str) -> set[str]:
        return {w.lower().strip(".,;:()") for w in text.split()}

    # Document frequency over the surface every index is required to index.
    df: dict[str, int] = {}
    for u in units:
        for w in words(u.indexing_text()):
            df[w] = df.get(w, 0) + 1

    probe: EnrichedUnit | None = None
    term = ""
    best_df = 0
    for u in units:
        ctx_text = u.indexing_text()[: -len(u.unit.text)] if u.unit.text else u.indexing_text()
        body_words = words(u.unit.text)
        for w in words(ctx_text):
            if len(w) > 5 and w not in body_words and (probe is None or df[w] < best_df):
                probe, term, best_df = u, w, df[w]
        if probe is not None and best_df == 1:
            break
    if probe is None:
        return ["cannot verify the retrieval surface: no enrichment adds a distinctive term"]

    result = index.search(IndexQuery(text=term, top_k=max(50, best_df)), ctx)
    if probe.unit_id not in result.unit_ids():
        problems.append(
            f"index {index.name!r} did not return the unit whose *context* contains "
            f"{term!r}. It is indexing unit.text rather than indexing_text(); the "
            f"contextualisation benefit is being lost on this index."
        )
    return problems


def check_ranked_list(rl: RankedList) -> list[str]:
    """The retrieval contract: dense 1-based ranks, provenance on every hit."""
    problems: list[str] = []
    for i, h in enumerate(rl.hits, start=1):
        if h.rank != i:
            problems.append(f"{rl.source}: hit at position {i} claims rank {h.rank}")
        if not h.provenance.document_id:
            problems.append(f"{rl.source}: hit {h.unit_id} has no document in its provenance")
    ids = [h.unit_id for h in rl.hits]
    if len(set(ids)) != len(ids):
        problems.append(f"{rl.source}: duplicate unit ids in one ranked list")
    return problems


def summarise(named: Mapping[str, Sequence[str]]) -> str:
    total = sum(len(v) for v in named.values())
    if not total:
        return "all contract checks passed"
    lines = [f"{total} contract violation(s):"]
    for name, problems in named.items():
        for p in problems:
            lines.append(f"  [{name}] {p}")
    return "\n".join(lines)


def check_context_specificity(
    units: Sequence[EnrichedUnit], *, max_overlap: float = 0.6
) -> list[str]:
    """Is the prepended context chunk-specific, or document-level boilerplate?

    Invariant 3's benefit comes from context that distinguishes one chunk from
    another. Context that is near-identical across a document's units is
    dilution wearing the same shape: it cannot help rank one unit above its
    sibling, while it does inflate length (BM25 penalises that) and pull every
    unit's dense vector toward the document centroid.

    Measured on a real corpus, an extractive contextualiser produced context
    that was 76% identical between adjacent units and occupied 27% of the
    indexed surface -- and retrieval got measurably worse. This check exists so
    that failure mode is a number rather than a mystery, because the symptom
    (contextualisation not helping) looks identical to the enricher being
    mis-wired, and the two have completely different fixes.
    """
    from indexer.textutil import tokenize

    by_doc: dict[str, list[EnrichedUnit]] = {}
    for eu in units:
        by_doc.setdefault(str(eu.document_id), []).append(eu)

    overlaps: list[float] = []
    shares: list[float] = []
    for group in by_doc.values():
        if len(group) < 2:
            continue
        contexts = []
        for eu in group:
            full, body = eu.indexing_text(), eu.unit.text
            ctx = full[: max(0, len(full) - len(body))]
            contexts.append(set(tokenize(ctx)))
            shares.append(len(ctx) / max(1, len(full)))
        for a, b in pairwise(contexts):
            if a or b:
                overlaps.append(len(a & b) / max(1, len(a | b)))

    if not overlaps:
        return []
    mean_overlap = sum(overlaps) / len(overlaps)
    mean_share = sum(shares) / len(shares) if shares else 0.0
    if mean_overlap <= max_overlap:
        return []
    return [
        f"context is {mean_overlap:.0%} identical between adjacent units of the same "
        f"document and occupies {mean_share:.0%} of the indexed surface. That portion "
        f"carries no signal for ranking one unit above its sibling, and it dilutes the "
        f"terms that do. Make the context describe the chunk, not the document."
    ]

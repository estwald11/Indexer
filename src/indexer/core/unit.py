"""Units: the atom of retrieval, and the enrichments attached to them.

A ``Unit`` is what `segment` produces and what every index stores. An
``EnrichedUnit`` is a unit plus whatever `enrich` attached to it.

The load-bearing method here is ``EnrichedUnit.indexing_text()``.

Invariant 3 says an LLM-written 50-100 token summary prepended before *both*
embedding *and* lexical indexing cuts top-20 retrieval failure from 5.7% to
2.9%. The failure mode that loses that benefit is mundane: the dense index gets
the contextualised string and the lexical index gets the raw one, because two
different implementations each decided what to index. So the frame does not let
them decide. ``indexing_text()`` is computed once, on the unit, and every index
is contractually required to use it as its retrieval surface. An index that
wants the raw text as well has ``unit.text``; an index that indexes *only*
``unit.text`` is out of contract and the ablation harness will show it.

That also makes "contextualisation off" a real ablation: with no ``context``
enrichment, ``indexing_text()`` degrades to ``unit.text`` and every index
follows automatically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from indexer.core.document import content_metadata
from indexer.core.ids import ContentHash, DocumentId, UnitId, hash_obj, hash_text
from indexer.core.provenance import Provenance

__all__ = [
    "ContextScope",
    "EnrichedUnit",
    "Enrichment",
    "FieldValue",
    "FieldValues",
    "Unit",
    "UnitKind",
]

#: Values admissible in the structured index. Deliberately narrow: these are the
#: types a predicate can compare, sort and aggregate over. Anything richer
#: belongs in ``Enrichment.extra`` and is not queryable structurally.
FieldValue = str | int | float | bool | date | datetime | None
#: A field may hold several values -- every VAT number a unit mentions, every
#: group an ACL grants. Predicates over it are existential.
FieldValues = FieldValue | tuple[FieldValue, ...]


class UnitKind(StrEnum):
    PROSE = "prose"
    TABLE = "table"
    TABLE_ROWS = "table_rows"
    FIGURE = "figure"
    CODE = "code"
    LIST = "list"
    MIXED = "mixed"


class ContextScope(StrEnum):
    """How much context an enricher reads -- and therefore what invalidates it.

    This is the knob that decides the cost of an incremental rebuild. A
    ``UNIT``-scoped enricher survives edits elsewhere in its document; a
    ``DOCUMENT``-scoped one (contextualisation, which is the whole point of
    invariant 3) is invalidated by any edit to the parent. Declaring the scope
    lets the cache key include exactly what was read and nothing more, so the
    blast radius of an edit is a property of the config, not a surprise.
    """

    UNIT = "unit"
    NEIGHBORS = "neighbors"
    DOCUMENT = "document"
    CORPUS = "corpus"


@dataclass(frozen=True, slots=True)
class Unit:
    """One indexable span of one document, with its structural address.

    Every unit carries its document, its section path and its source span --
    that is the segment contract, and it is why a hit can always be cited.
    """

    unit_id: UnitId
    document_id: DocumentId
    text: str
    provenance: Provenance
    #: Heading trail from document root, outermost first. Empty for documents
    #: with no headings; never ``None``, so consumers need no null branch.
    section_path: tuple[str, ...] = field(default_factory=tuple)
    kind: UnitKind | str = UnitKind.PROSE
    #: Position in reading order within the document. Used for neighbour
    #: windows and for stable ``occurrence`` assignment -- never for identity.
    ordinal: int = 0
    prev_unit_id: UnitId | None = None
    next_unit_id: UnitId | None = None
    #: Structure preserved from the parsed block, when the unit is one table.
    table_ref: str | None = None
    #: Whether ``text`` is the document's canonical text at ``span``, verbatim.
    #:
    #: Almost always true, and the frame checks it, because a unit whose text
    #: is not at the span it claims produces a citation pointing at the wrong
    #: place. The documented exception is a table split by row groups with its
    #: header rows repeated into each piece: the text is then a *derivation* of
    #: the span rather than a copy of it. Making that a declared flag rather
    #: than a tolerance in the checker means the exception is visible in the
    #: data, auditable, and cannot quietly widen to cover real bugs.
    verbatim: bool = True
    #: Document-level metadata copied down so an index can filter without a
    #: join. Scanner-supplied facts only; inferred fields live in enrichments.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> ContentHash:
        return hash_text(self.text)

    @property
    def section_heading(self) -> str:
        return self.section_path[-1] if self.section_path else ""


@dataclass(frozen=True, slots=True)
class Enrichment:
    """One enricher's output for one unit.

    The three typed payloads correspond to the three things enrich is for:

    ``context``
        Prose prepended to the retrieval surface. Invariant 3.
    ``fields``
        Typed values written to the structured index. Invariant 5 -- this is
        what lets a numeric or temporal question bypass vector search entirely.
    ``labels``
        Classifications used by the router and by filters.

    ``extra`` carries anything else; the frame stores and hashes it but never
    interprets it.
    """

    enricher: str
    fingerprint: str
    context: str | None = None
    fields: Mapping[str, FieldValues] = field(default_factory=dict)
    labels: Mapping[str, str | tuple[str, ...]] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    #: What the enricher read. Determines the cache key, hence invalidation.
    scope: ContextScope = ContextScope.UNIT
    #: Accounting for this one enrichment, aggregated into the manifest.
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class EnrichedUnit:
    """A unit plus its enrichments. The input type of every index."""

    unit: Unit
    #: Keyed by enricher name, so a second run of the same enricher replaces
    #: rather than duplicates -- which is what makes enrich idempotent.
    enrichments: Mapping[str, Enrichment] = field(default_factory=dict)

    @property
    def unit_id(self) -> UnitId:
        return self.unit.unit_id

    @property
    def document_id(self) -> DocumentId:
        return self.unit.document_id

    def indexing_text(self) -> str:
        """The retrieval surface. Every index MUST index this string.

        Context blocks come first, in stable enricher-name order, then the
        unit's own text. Stable order matters: the string is content-hashed,
        and a dict-iteration-order dependency would produce cache misses that
        look like corruption.

        With no context enrichments this returns ``unit.text`` unchanged, which
        is exactly the "contextualisation off" arm of the ablation.
        """
        parts = [
            e.context.strip()
            for _, e in sorted(self.enrichments.items())
            if e.context and e.context.strip()
        ]
        parts.append(self.unit.text)
        return "\n\n".join(parts)

    @property
    def indexing_hash(self) -> ContentHash:
        """Hash of the retrieval surface. Decides whether a vector is recomputed."""
        return hash_text(self.indexing_text())

    @property
    def record_hash(self) -> ContentHash:
        """Hash of everything an index may store for this unit.

        The surface is not enough to decide that a stored unit is current. A
        paragraph inserted above a unit moves its span without touching its
        text or id; a corrected extraction rule changes its fields without
        touching its surface. Indexes that skipped a write because the surface
        hash matched kept the old span -- every citation from the unit pointing
        at the wrong offset -- and the old field values, so a fixed extractor
        appeared to do nothing. The surface hash still decides whether a
        *vector* is recomputed; this decides whether the record is rewritten.

        Position within the document (ordinal, neighbour links, block ids) is
        deliberately left out: no index stores it, it changes for every unit
        below an edit, and the unit store -- which does hold it -- compares its
        own full encoding instead.
        """
        u, p = self.unit, self.unit.provenance
        return hash_obj(
            {
                "surface": self.indexing_text(),
                "fields": _typed(self.fields()),
                "labels": _typed(self.labels()),
                "document_id": u.document_id,
                "span": [p.span.start, p.span.end],
                "source_uri": p.source_uri,
                "pages": [p.pages.start, p.pages.end] if p.pages else None,
                "section_path": list(u.section_path),
                "kind": str(u.kind),
                "table_ref": u.table_ref,
                "metadata": _typed(content_metadata(u.metadata)),
            }
        )

    def fields(self) -> dict[str, FieldValues]:
        """All extracted fields, flattened. Later enrichers win on collision.

        Collisions are resolved by sorted enricher name rather than run order so
        the result is deterministic; a config with two enrichers writing the
        same field is a config smell the validator warns about.
        """
        out: dict[str, FieldValues] = {}
        for _, e in sorted(self.enrichments.items()):
            out.update(e.fields)
        return out

    def filter_fields(self) -> dict[str, Any]:
        """What filters and structured queries see: extracted fields, overlaid
        by the scanner's metadata.

        Scanner metadata used to reach an index only if an enricher copied it
        (``regex_fields.from_metadata``), so a tenant or ACL filter silently
        excluded everything in any config that forgot the copy. It now reaches
        every index directly.

        Metadata wins a name collision, deliberately. It states known facts --
        the tenant a folder belongs to, the groups a sidecar grants -- while
        fields are inferred from content, and a document must not be able to
        re-scope itself by containing text an extractor reads as ``tenant``.
        Lists of scalars (an ACL) are kept as tuples; nested structures are
        not filterable and are left out.
        """
        out: dict[str, Any] = dict(self.fields())
        for k, v in content_metadata(self.unit.metadata).items():
            if isinstance(v, (list, tuple, set, frozenset)):
                items = sorted(v, key=repr) if isinstance(v, (set, frozenset)) else list(v)
                if all(_is_scalar(x) for x in items):
                    out[k] = tuple(items)
            elif _is_scalar(v) and v is not None:
                out[k] = v
        return out

    def labels(self) -> dict[str, str | tuple[str, ...]]:
        out: dict[str, str | tuple[str, ...]] = {}
        for _, e in sorted(self.enrichments.items()):
            out.update(e.labels)
        return out

    def with_enrichment(self, enrichment: Enrichment) -> EnrichedUnit:
        """Attach or replace one enricher's output. Idempotent by construction."""
        merged = dict(self.enrichments)
        merged[enrichment.enricher] = enrichment
        return replace(self, enrichments=merged)

    def cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.enrichments.values())


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool, date, datetime))


def _typed(v: Any) -> Any:
    """A hashable rendering that keeps the type: ``date(2024,1,1)`` and the
    string ``"2024-01-01"`` must not hash alike, or an extractor fixed to emit a
    date instead of a string would leave the string in every index."""
    if isinstance(v, bool):
        return ["bool", v]
    if isinstance(v, datetime):
        return ["datetime", v.isoformat()]
    if isinstance(v, date):
        return ["date", v.isoformat()]
    if v is None or isinstance(v, (int, float, str)):
        return [type(v).__name__, v]
    if isinstance(v, Mapping):
        return ["map", {str(k): _typed(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}]
    if isinstance(v, (list, tuple, set, frozenset)):
        items = sorted(v, key=repr) if isinstance(v, (set, frozenset)) else list(v)
        return ["list", [_typed(x) for x in items]]
    return ["repr", repr(v)]


def bare(units: Sequence[Unit]) -> list[EnrichedUnit]:
    """Lift units to enriched units with no enrichments.

    The identity behaviour of a disabled `enrich` stage. Named so that the
    disabled path is a deliberate call rather than an implicit construction.
    """
    return [EnrichedUnit(unit=u) for u in units]

"""JSON codecs for the types that cross a cache boundary.

The cache stores bytes, not objects, so every stage boundary needs an explicit
serialisation. Explicit rather than pickle: a pickle cache turns any dataclass
change into either a silent corpus-wide invalidation or an unpickling error
halfway through a build, and neither is discoverable at the point of the change.

Being explicit also means the cache is inspectable. When a build produces
something unexpected, ``cat``-ing a cache entry and reading it is the fastest
diagnosis available, and that is worth more than the few minutes this file costs.
"""

from __future__ import annotations

from typing import Any

from indexer.core.document import Block, BlockKind, MediaRef, ParsedDocument, Table, TableCell
from indexer.core.ids import ContentHash, DocumentId, UnitId
from indexer.core.provenance import BBox, PageRef, Provenance, Span
from indexer.core.unit import ContextScope, EnrichedUnit, Enrichment, Unit, UnitKind

__all__ = [
    "decode_enriched_unit",
    "decode_parsed_document",
    "decode_units",
    "encode_enriched_unit",
    "encode_parsed_document",
    "encode_units",
]


def _enc_prov(p: Provenance) -> dict[str, Any]:
    return {
        "d": p.document_id,
        "s": [p.span.start, p.span.end],
        "u": p.source_uri,
        "p": [p.pages.start, p.pages.end] if p.pages else None,
        "b": [p.bbox.x0, p.bbox.y0, p.bbox.x1, p.bbox.y1] if p.bbox else None,
        "ss": [p.source_span.start, p.source_span.end] if p.source_span else None,
        "bi": list(p.block_ids),
    }


def _dec_prov(d: dict[str, Any]) -> Provenance:
    return Provenance(
        document_id=DocumentId(d["d"]),
        span=Span(d["s"][0], d["s"][1]),
        source_uri=d.get("u", ""),
        pages=PageRef(d["p"][0], d["p"][1]) if d.get("p") else None,
        bbox=BBox(*d["b"]) if d.get("b") else None,
        source_span=Span(d["ss"][0], d["ss"][1]) if d.get("ss") else None,
        block_ids=tuple(d.get("bi", ())),
    )


def encode_parsed_document(doc: ParsedDocument) -> dict[str, Any]:
    return {
        "document_id": doc.document_id,
        "source_uri": doc.source_uri,
        "text": doc.text,
        "source_hash": doc.source_hash,
        "page_count": doc.page_count,
        "reading_order_confidence": doc.reading_order_confidence,
        "metadata": dict(doc.metadata),
        "page_media": [
            {"uri": m.uri, "mt": m.media_type, "h": m.content_hash, "w": m.width, "ht": m.height}
            for m in doc.page_media
        ],
        "blocks": [
            {
                "id": b.block_id,
                "k": str(b.kind),
                "t": b.text,
                "p": _enc_prov(b.provenance),
                "l": b.level,
                "tb": (
                    {
                        "rows": [
                            [[c.text, c.row_span, c.col_span, c.is_header] for c in row]
                            for row in b.table.rows
                        ],
                        "cap": b.table.caption,
                        "hr": b.table.header_rows,
                    }
                    if b.table
                    else None
                ),
                "m": (
                    {
                        "uri": b.media.uri,
                        "mt": b.media.media_type,
                        "h": b.media.content_hash,
                        "w": b.media.width,
                        "ht": b.media.height,
                    }
                    if b.media
                    else None
                ),
                "a": dict(b.attrs),
            }
            for b in doc.blocks
        ],
    }


def decode_parsed_document(d: dict[str, Any]) -> ParsedDocument:
    return ParsedDocument(
        document_id=DocumentId(d["document_id"]),
        source_uri=d["source_uri"],
        text=d["text"],
        source_hash=ContentHash(d["source_hash"]),
        page_count=d.get("page_count"),
        reading_order_confidence=d.get("reading_order_confidence", 1.0),
        metadata=d.get("metadata", {}),
        page_media=tuple(
            MediaRef(m["uri"], m["mt"], ContentHash(m["h"]), m.get("w"), m.get("ht"))
            for m in d.get("page_media", ())
        ),
        blocks=tuple(
            Block(
                block_id=b["id"],
                kind=_block_kind(b["k"]),
                text=b["t"],
                provenance=_dec_prov(b["p"]),
                level=b.get("l"),
                table=(
                    Table(
                        rows=tuple(
                            tuple(TableCell(c[0], c[1], c[2], c[3]) for c in row)
                            for row in b["tb"]["rows"]
                        ),
                        caption=b["tb"].get("cap"),
                        header_rows=b["tb"].get("hr", 0),
                    )
                    if b.get("tb")
                    else None
                ),
                media=(
                    MediaRef(
                        b["m"]["uri"],
                        b["m"]["mt"],
                        ContentHash(b["m"]["h"]),
                        b["m"].get("w"),
                        b["m"].get("ht"),
                    )
                    if b.get("m")
                    else None
                ),
                attrs=b.get("a", {}),
            )
            for b in d["blocks"]
        ),
    )


def _block_kind(s: str) -> BlockKind | str:
    # Unknown kinds round-trip as plain strings rather than raising: BlockKind is
    # open by convention, and a cache that refuses a parser's new kind would make
    # adding one a cache-clearing event.
    try:
        return BlockKind(s)
    except ValueError:
        return s


def _enc_unit(u: Unit) -> dict[str, Any]:
    return {
        "id": u.unit_id,
        "d": u.document_id,
        "t": u.text,
        "p": _enc_prov(u.provenance),
        "sp": list(u.section_path),
        "k": str(u.kind),
        "o": u.ordinal,
        "pv": u.prev_unit_id,
        "nx": u.next_unit_id,
        "tr": u.table_ref,
        "vb": u.verbatim,
        "m": dict(u.metadata),
    }


def _dec_unit(d: dict[str, Any]) -> Unit:
    try:
        kind: UnitKind | str = UnitKind(d.get("k", "prose"))
    except ValueError:
        kind = d.get("k", "prose")
    return Unit(
        unit_id=UnitId(d["id"]),
        document_id=DocumentId(d["d"]),
        text=d["t"],
        provenance=_dec_prov(d["p"]),
        section_path=tuple(d.get("sp", ())),
        kind=kind,
        ordinal=d.get("o", 0),
        prev_unit_id=UnitId(d["pv"]) if d.get("pv") else None,
        next_unit_id=UnitId(d["nx"]) if d.get("nx") else None,
        table_ref=d.get("tr"),
        verbatim=d.get("vb", True),
        metadata=d.get("m", {}),
    )


def encode_units(units: list[Unit]) -> list[dict[str, Any]]:
    return [_enc_unit(u) for u in units]


def decode_units(ds: list[dict[str, Any]]) -> list[Unit]:
    return [_dec_unit(d) for d in ds]


def _enc_enrichment(e: Enrichment) -> dict[str, Any]:
    return {
        "e": e.enricher,
        "f": e.fingerprint,
        "c": e.context,
        "fl": {k: _enc_scalar(v) for k, v in e.fields.items()},
        "l": {k: list(v) if isinstance(v, tuple) else v for k, v in e.labels.items()},
        "x": dict(e.extra),
        "s": str(e.scope),
        "ti": e.tokens_in,
        "to": e.tokens_out,
        "cu": e.cost_usd,
    }


def _enc_scalar(v: Any) -> Any:
    # Dates carry a type tag so they survive the round trip as dates. Losing the
    # tag would turn every temporal predicate into a string comparison, which
    # compares "2023-1-5" against "2023-11-02" wrongly and silently.
    if hasattr(v, "isoformat"):
        return {"__t": "datetime" if hasattr(v, "hour") else "date", "v": v.isoformat()}
    return v


def _dec_scalar(v: Any) -> Any:
    if isinstance(v, dict) and "__t" in v:
        from datetime import date, datetime

        return (
            datetime.fromisoformat(v["v"]) if v["__t"] == "datetime" else date.fromisoformat(v["v"])
        )
    return v


def _dec_enrichment(d: dict[str, Any]) -> Enrichment:
    return Enrichment(
        enricher=d["e"],
        fingerprint=d.get("f", ""),
        context=d.get("c"),
        fields={k: _dec_scalar(v) for k, v in d.get("fl", {}).items()},
        labels={k: tuple(v) if isinstance(v, list) else v for k, v in d.get("l", {}).items()},
        extra=d.get("x", {}),
        scope=ContextScope(d.get("s", "unit")),
        tokens_in=d.get("ti", 0),
        tokens_out=d.get("to", 0),
        cost_usd=d.get("cu", 0.0),
    )


def encode_enriched_unit(eu: EnrichedUnit) -> dict[str, Any]:
    return {
        "u": _enc_unit(eu.unit),
        "e": {k: _enc_enrichment(v) for k, v in eu.enrichments.items()},
    }


def decode_enriched_unit(d: dict[str, Any]) -> EnrichedUnit:
    return EnrichedUnit(
        unit=_dec_unit(d["u"]),
        enrichments={k: _dec_enrichment(v) for k, v in d.get("e", {}).items()},
    )

"""Identity and content addressing.

Two different notions of identity run through the whole frame, and conflating
them is the single easiest way to break incremental rebuilds:

``DocumentId``
    *Which* document this is. Stable across edits. Derived from the source URI
    (or supplied by the caller), never from content. A contract that is edited
    keeps its id.

``ContentHash``
    *What* the bytes are. Changes on every edit. Every artifact at every stage
    boundary carries one, and every cache key is built from one.

The pair is what makes "adding 10 documents to 500 reprocesses 10" a property of
the frame rather than a property of any implementation: the ledger compares the
``DocumentId`` to find the prior state and the ``ContentHash`` to decide whether
that state is still valid.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, NewType

__all__ = [
    "BuildId",
    "ContentHash",
    "DocumentId",
    "UnitId",
    "canonical_json",
    "hash_bytes",
    "hash_obj",
    "hash_text",
    "make_document_id",
    "make_unit_id",
    "merge_hashes",
    "short",
]

DocumentId = NewType("DocumentId", str)
UnitId = NewType("UnitId", str)
ContentHash = NewType("ContentHash", str)
BuildId = NewType("BuildId", str)

_ALGO = "sha256"
_PREFIX = f"{_ALGO}:"


def hash_bytes(data: bytes) -> ContentHash:
    """Content hash of raw bytes. The only hash primitive in the frame."""
    return ContentHash(_PREFIX + hashlib.sha256(data).hexdigest())


def hash_text(text: str) -> ContentHash:
    """Content hash of text. UTF-8, NFC-insensitive: we hash exactly what we got.

    Normalisation is a *parse* concern. Hashing normalised text here would make
    two stages disagree about what the content is.
    """
    return hash_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding, the canonical form for hashing structures.

    Sorted keys, no insignificant whitespace, UTF-8. Two configs that differ only
    in key order MUST produce the same hash, or every reordering of a YAML file
    would invalidate an entire index.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_fallback
    ).encode("utf-8")


def _fallback(obj: Any) -> Any:
    # Keep the set of encodable things small and explicit. An un-encodable value
    # in a params dict is a bug in the caller, not something to paper over: it
    # would silently produce two different hashes for the same configuration.
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return list(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"{type(obj).__name__} is not canonically encodable; give it an explicit form")


def hash_obj(obj: Any) -> ContentHash:
    """Content hash of a JSON-encodable structure (params, configs, manifests)."""
    return hash_bytes(canonical_json(obj))


def merge_hashes(*parts: str) -> ContentHash:
    """Combine hashes and discriminators into one.

    Length-prefixed so that ``("ab", "c")`` and ``("a", "bc")`` cannot collide --
    cache keys are built from this, and a collision would serve one stage's
    output as another's.
    """
    h = hashlib.sha256()
    for p in parts:
        raw = p.encode("utf-8")
        h.update(len(raw).to_bytes(8, "big"))
        h.update(raw)
    return ContentHash(_PREFIX + h.hexdigest())


def short(h: str, length: int = 12) -> str:
    """Display form. Never use in a key: truncation raises collision risk."""
    return h[len(_PREFIX) :][:length] if h.startswith(_PREFIX) else h[:length]


def make_document_id(source_uri: str, *, namespace: str = "") -> DocumentId:
    """Derive a stable document id from its location.

    Deliberately *not* content-derived. Callers with a real primary key (a case
    number, a ticket id) should pass it through instead of calling this.
    """
    return DocumentId(short(merge_hashes("doc", namespace, source_uri), 24))


def make_unit_id(
    document_id: DocumentId, content_hash: ContentHash, *, occurrence: int = 0
) -> UnitId:
    """Derive a unit id from its document and its own content.

    Position is *not* an input: inserting a paragraph on page 1 must not change
    the id of every unit after it, or an incremental rebuild would rewrite the
    whole document's worth of vectors for a one-line edit. ``occurrence``
    disambiguates genuinely identical text within one document (repeated
    boilerplate, table headers) and is assigned in reading order.
    """
    return UnitId(short(merge_hashes("unit", document_id, content_hash, str(occurrence)), 24))

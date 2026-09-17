"""Filesystem-backed stores: cache, ledger, unit store.

Reference implementations of the storage protocols. Deliberately simple -- a
directory of files, JSON on disk -- because the interesting properties are
correctness ones (a cache that survives a restart, a ledger that records
deletions) and those are testable without a database.

The seam to something better is the protocol. A Postgres ledger and an S3 cache
are drop-ins; nothing above them knows the difference.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from indexer.core.cache import CacheKey
from indexer.core.ids import BuildId, ContentHash, DocumentId, UnitId, hash_bytes, short
from indexer.core.ledger import DocumentRecord
from indexer.core.unit import EnrichedUnit
from indexer.io import atomic_write

__all__ = ["FileArtifactStore", "FileCache", "JsonLedger", "UnitStore", "atomic_write"]


class FileCache:
    """Content-addressed cache over a directory.

    Keys are hashes, so they fan out into two levels of subdirectory -- a flat
    directory with a million entries is slow on most filesystems and unpleasant
    to inspect.
    """

    __slots__ = ("root",)

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: CacheKey) -> Path:
        h = short(key, 64)
        return self.root / h[:2] / h[2:4] / h

    def get(self, key: CacheKey) -> bytes | None:
        p = self._path(key)
        try:
            return p.read_bytes()
        except FileNotFoundError:
            return None

    def put(self, key: CacheKey, value: bytes) -> None:
        atomic_write(self._path(key), value)

    def has(self, key: CacheKey) -> bool:
        return self._path(key).exists()

    def delete(self, key: CacheKey) -> None:
        self._path(key).unlink(missing_ok=True)

    def iter_keys(self, prefix: str = "") -> Iterator[CacheKey]:
        if not self.root.exists():
            return
        for p in self.root.rglob("*"):
            if p.is_file() and p.name.startswith(prefix):
                yield p.name


class FileArtifactStore:
    """Large binaries by content hash.

    Separate from the cache because the lifetimes differ: evicting a cache entry
    costs recomputation, evicting an artifact breaks a ``MediaRef`` held in an
    index and therefore breaks provenance.
    """

    __slots__ = ("root",)

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, h: ContentHash | str) -> Path:
        s = short(str(h), 64)
        return self.root / s[:2] / s

    def put(self, content: bytes, media_type: str) -> str:
        h = hash_bytes(content)
        p = self._path(h)
        if not p.exists():
            atomic_write(p, content)
        return f"indexer://artifact/{h}?type={media_type}"

    def get(self, uri: str) -> bytes:
        h = uri.removeprefix("indexer://artifact/").split("?", 1)[0]
        return self._path(h).read_bytes()

    def exists(self, uri: str) -> bool:
        h = uri.removeprefix("indexer://artifact/").split("?", 1)[0]
        return self._path(h).exists()


class JsonLedger:
    """Per-document build state as one JSON file.

    One file, not one per document: a build's plan reads every record, and a
    thousand small reads is slower than one. The cost is that concurrent builds
    would race -- which is the documented seam where a distributed build needs a
    real database, noted in ARCHITECTURE.md.
    """

    __slots__ = ("_in_flight", "_loaded", "_records", "path")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, DocumentRecord] = {}
        self._in_flight: str | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self._in_flight = data.get("in_flight")
        for d in data.get("records", []):
            rec = DocumentRecord(
                document_id=DocumentId(d["document_id"]),
                source_uri=d["source_uri"],
                content_hash=ContentHash(d["content_hash"]),
                unit_ids=tuple(UnitId(u) for u in d["unit_ids"]),
                stage_keys=d["stage_keys"],
                build_id=BuildId(d["build_id"]),
                updated_at=d.get("updated_at", ""),
                warnings=tuple(d.get("warnings", ())),
            )
            self._records[rec.document_id] = rec

    def _flush(self) -> None:
        payload = {
            "in_flight": self._in_flight,
            "records": [
                {
                    "document_id": r.document_id,
                    "source_uri": r.source_uri,
                    "content_hash": r.content_hash,
                    "unit_ids": list(r.unit_ids),
                    "stage_keys": dict(r.stage_keys),
                    "build_id": r.build_id,
                    "updated_at": r.updated_at,
                    "warnings": list(r.warnings),
                }
                for r in self._records.values()
            ],
        }
        atomic_write(self.path, json.dumps(payload, indent=1).encode("utf-8"))

    def get(self, document_id: DocumentId) -> DocumentRecord | None:
        self._load()
        return self._records.get(document_id)

    def put(self, record: DocumentRecord) -> None:
        self._load()
        self._records[record.document_id] = record
        # Flushed per document rather than at the end: an interrupted build must
        # resume at document 401, not restart. Resumability is a stated property.
        self._flush()

    def delete(self, document_id: DocumentId) -> None:
        self._load()
        self._records.pop(document_id, None)
        self._flush()

    def iter_records(self) -> Iterator[DocumentRecord]:
        self._load()
        yield from list(self._records.values())

    def document_ids(self) -> frozenset[DocumentId]:
        self._load()
        return frozenset(DocumentId(k) for k in self._records)

    def begin_build(self, build_id: BuildId) -> None:
        self._load()
        self._in_flight = build_id
        self._flush()

    def commit_build(self, build_id: BuildId) -> None:
        self._load()
        self._in_flight = None
        self._flush()

    @property
    def interrupted_build(self) -> str | None:
        self._load()
        return self._in_flight


class UnitStore:
    """The units themselves, so hits can be hydrated and passages returned.

    Separate from every index on purpose. An index may store only ids and
    vectors -- that is a legitimate index -- but a returned passage needs text
    and provenance. Making each index store the full unit would duplicate the
    corpus once per index and make "add a fourth index" expensive in a way the
    design promises it is not.
    """

    __slots__ = ("_loaded", "_units", "path")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._units: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path.exists():
            self._units = json.loads(self.path.read_text())

    def put_many(self, units: list[EnrichedUnit]) -> None:
        from indexer.pipeline.codec import encode_enriched_unit

        self._load()
        for u in units:
            self._units[u.unit_id] = encode_enriched_unit(u)
        self._flush()

    def get(self, unit_id: UnitId) -> EnrichedUnit | None:
        from indexer.pipeline.codec import decode_enriched_unit

        self._load()
        raw = self._units.get(unit_id)
        return decode_enriched_unit(raw) if raw else None

    def delete_many(self, unit_ids: list[UnitId]) -> int:
        self._load()
        n = sum(1 for u in unit_ids if self._units.pop(u, None) is not None)
        if n:
            self._flush()
        return n

    def all_ids(self) -> list[UnitId]:
        self._load()
        return [UnitId(k) for k in self._units]

    def __len__(self) -> int:
        self._load()
        return len(self._units)

    def _flush(self) -> None:
        atomic_write(self.path, json.dumps(self._units).encode("utf-8"))

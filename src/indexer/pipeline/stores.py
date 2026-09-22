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
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from indexer.core.cache import CacheKey
from indexer.core.ids import BuildId, ContentHash, DocumentId, UnitId, hash_bytes, short
from indexer.core.ledger import DocumentRecord
from indexer.core.unit import EnrichedUnit
from indexer.io import atomic_write

__all__ = [
    "CacheRefs",
    "FileArtifactStore",
    "FileCache",
    "JsonLedger",
    "UnitStore",
    "atomic_write",
    "sweep_cache",
]


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


class CacheRefs:
    """Which cache entries each document's current build uses.

    The cache is content-addressed and deliberately shared -- one parse serves
    every copy of a file -- so "delete this document's cache entries" needs to
    know both which entries it used and whether anything else still uses them.
    Without this, removing a document removed its units from every index and
    left its full text, its LLM-written summaries and its extracted fields in
    the cache indefinitely: an erasure request that could not be honoured.

    SQLite rather than a column in the ledger: a document uses one entry per
    unit per enricher, and a JSON ledger carrying those for a large archive
    would be rewritten -- all of it -- at every compaction.
    """

    __slots__ = ("_conn", "path")

    def __init__(self, path: str | Path) -> None:
        import sqlite3

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS refs (
                document_id TEXT NOT NULL,
                key         TEXT NOT NULL,
                PRIMARY KEY (document_id, key)
            );
            CREATE INDEX IF NOT EXISTS ix_refs_key ON refs(key);
            """
        )
        self._conn.commit()

    def keys_of(self, document_id: str) -> set[str]:
        return {
            r[0]
            for r in self._conn.execute(
                "SELECT key FROM refs WHERE document_id = ?", (document_id,)
            )
        }

    def record(self, document_id: str, keys: set[str]) -> set[str]:
        """Record a document's current keys; return the ones it no longer uses."""
        old = self.keys_of(document_id)
        self._conn.execute("DELETE FROM refs WHERE document_id = ?", (document_id,))
        self._conn.executemany(
            "INSERT OR IGNORE INTO refs (document_id, key) VALUES (?, ?)",
            [(document_id, k) for k in sorted(keys)],
        )
        self._conn.commit()
        return old - keys

    def remove(self, document_id: str) -> set[str]:
        """Forget a document; return every key it used."""
        old = self.keys_of(document_id)
        self._conn.execute("DELETE FROM refs WHERE document_id = ?", (document_id,))
        self._conn.commit()
        return old

    def unreferenced(self, keys: set[str]) -> set[str]:
        """The subset of ``keys`` no document uses."""
        out: set[str] = set()
        for k in keys:
            if self._conn.execute("SELECT 1 FROM refs WHERE key = ? LIMIT 1", (k,)).fetchone():
                continue
            out.add(k)
        return out

    def all_keys(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT DISTINCT key FROM refs")}


def sweep_cache(cache: Any, keep: set[str]) -> int:
    """Delete every cache entry not in ``keep``. The full, mark-and-sweep GC.

    For entries no build recorded -- written by a build that crashed before its
    checkpoint, or by a version that did not record references. Compared on the
    digest, because a file cache names entries by digest rather than full key.
    """
    wanted = {short(k, 64) for k in keep}
    doomed = [k for k in cache.iter_keys() if short(k, 64) not in wanted]
    for k in doomed:
        cache.delete(k)
    return len(doomed)


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
    """Per-document build state: a compacted snapshot plus an append-only journal.

    One file, not one per document: a build's plan reads every record, and a
    thousand small reads is slower than one. The cost is that concurrent builds
    would race -- the documented seam where a distributed build needs a real
    database, noted in ARCHITECTURE.md.

    Why a journal. Resumability is a stated property: a build that dies after
    400 of 500 documents must resume at 401, so a record has to be durable the
    moment its document is done. Rewriting the whole snapshot to achieve that is
    quadratic -- 686 documents against a 479 KB map cost 9s of a 30s build, a
    third of it, to write 328 MB for 479 KB of data.

    So each record is appended as one line (bounded, proportional to the record)
    and the snapshot is compacted once at ``commit_build``. A load reads the
    snapshot and replays the journal over it, so an interrupted build is
    recovered exactly -- the durability guarantee is unchanged and only its cost
    is different.
    """

    __slots__ = ("_in_flight", "_journal_lines", "_loaded", "_records", "journal", "path")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.journal = self.path.with_suffix(self.path.suffix + ".journal")
        self._records: dict[str, DocumentRecord] = {}
        self._in_flight: str | None = None
        self._loaded = False
        self._journal_lines = 0

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self._in_flight = data.get("in_flight")
        for d in [*data.get("records", []), *self._replay_journal()]:
            if d.get("deleted"):
                self._records.pop(d["document_id"], None)
                continue
            rec = DocumentRecord(
                document_id=DocumentId(d["document_id"]),
                source_uri=d["source_uri"],
                content_hash=ContentHash(d["content_hash"]),
                unit_ids=tuple(UnitId(u) for u in d["unit_ids"]),
                stage_keys=d["stage_keys"],
                build_id=BuildId(d["build_id"]),
                updated_at=d.get("updated_at", ""),
                warnings=tuple(d.get("warnings", ())),
                metadata_hash=d.get("metadata_hash", ""),
            )
            self._records[rec.document_id] = rec

    def _replay_journal(self) -> list[dict[str, Any]]:
        """Records written since the last compaction. Order matters: later wins."""
        if not self.journal.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line means the process died mid-append. Everything
                # before it is intact, and the document it described will simply
                # be reprocessed -- which is the correct outcome, not an error.
                break
        return out

    def _append(self, record: DocumentRecord | dict[str, Any]) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        line = record if isinstance(record, dict) else _record_json(record)
        with self.journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._journal_lines += 1

    def _flush(self) -> None:
        payload = {
            "in_flight": self._in_flight,
            "records": [_record_json(r) for r in self._records.values()],
        }
        atomic_write(self.path, json.dumps(payload, indent=1).encode("utf-8"))

    def get(self, document_id: DocumentId) -> DocumentRecord | None:
        self._load()
        return self._records.get(document_id)

    def put(self, record: DocumentRecord) -> None:
        self._load()
        self._records[record.document_id] = record
        # Appended per document, not rewritten: an interrupted build must resume
        # at document 401, not restart, and durability per document is what buys
        # that. The append is bounded; the snapshot is compacted at commit.
        self._append(record)

    def delete(self, document_id: DocumentId) -> None:
        self._load()
        self._records.pop(document_id, None)
        # A tombstone in the journal, not a snapshot rewrite: removing a folder
        # of 5,000 documents rewrote the whole ledger 5,000 times.
        self._append({"document_id": document_id, "deleted": True})

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
        # Compaction point: the snapshot now contains everything the journal did.
        self.journal.unlink(missing_ok=True)
        self._journal_lines = 0

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

    __slots__ = ("_dirty", "_loaded", "_units", "path")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._units: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._dirty = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path.exists():
            self._units = json.loads(self.path.read_text())

    def put_many(self, units: list[EnrichedUnit]) -> None:
        self._load()
        for u in units:
            encoded, rh = _encode_with_hash(u)
            self._units[u.unit_id] = {**encoded, "rh": rh}
        # Buffered, not written. Writing the whole map per document is quadratic
        # in corpus size -- the same defect the indexes had, and it hid here
        # longer because the unit store is not an Index and so was not covered
        # by the Flushable sweep. The pipeline commits once per build.
        self._dirty = True

    def is_current(self, unit: EnrichedUnit) -> bool:
        """Whether the stored record for this unit id is exactly this unit.

        An unchanged id does not mean an unchanged unit: ids are derived from
        text, so a unit keeps its id when a paragraph above it moves its span,
        when its heading is renamed, or when an enricher's output changes. The
        pipeline used to rewrite only new ids, and those units kept their old
        spans and section paths in every store.
        """
        self._load()
        raw = self._units.get(unit.unit_id)
        return raw is not None and raw.get("rh") == _encode_with_hash(unit)[1]

    def get(self, unit_id: UnitId) -> EnrichedUnit | None:
        from indexer.pipeline.codec import decode_enriched_unit

        self._load()
        raw = self._units.get(unit_id)
        return decode_enriched_unit(raw) if raw else None

    def delete_many(self, unit_ids: list[UnitId]) -> int:
        self._load()
        n = sum(1 for u in unit_ids if self._units.pop(u, None) is not None)
        if n:
            self._dirty = True
        return n

    def all_ids(self) -> list[UnitId]:
        self._load()
        return [UnitId(k) for k in self._units]

    def __len__(self) -> int:
        self._load()
        return len(self._units)

    def flush(self) -> None:
        """Persist. Called once per build by the ingestion pipeline."""
        if self._dirty:
            atomic_write(self.path, json.dumps(self._units).encode("utf-8"))
            self._dirty = False


def _encode_with_hash(u: EnrichedUnit) -> tuple[dict[str, Any], str]:
    from indexer.pipeline.codec import encode_enriched_unit

    encoded = encode_enriched_unit(u)
    return encoded, hash_bytes(json.dumps(encoded, sort_keys=True).encode("utf-8"))


def _record_json(r: DocumentRecord) -> dict[str, Any]:
    return {
        "document_id": r.document_id,
        "source_uri": r.source_uri,
        "content_hash": r.content_hash,
        "unit_ids": list(r.unit_ids),
        "stage_keys": dict(r.stage_keys),
        "build_id": r.build_id,
        "updated_at": r.updated_at,
        "warnings": list(r.warnings),
        "metadata_hash": r.metadata_hash,
    }

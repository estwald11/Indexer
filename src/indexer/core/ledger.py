"""The ledger: what makes "add 10 documents to 500, reprocess 10" a guarantee.

Incrementality is usually attempted at the cache layer alone, and that is not
enough. A cache tells you whether a *given* computation has been done. It cannot
tell you that a document was **deleted** from the corpus, so its units linger in
the index forever; nor which units a document previously produced, so a document
that shrinks leaves orphans that still answer queries.

The ledger is the missing side: a per-document record of what the last build
produced, and under which stage fingerprints. A build then becomes a diff.

    for each document now in the corpus:
        no ledger entry            -> ADDED     (full process)
        content hash differs       -> CHANGED   (full process, then diff units)
        fingerprints differ        -> RESTAGED  (reprocess affected stages only)
        otherwise                  -> UNCHANGED (skip entirely)
    entries with no document       -> REMOVED   (delete their units from indexes)

``RESTAGED`` is why fingerprints are stored per stage rather than as one blob:
changing the reranker must not reprocess anything, changing the segmenter must
reprocess segment onward, and changing one enricher must rerun that enricher
only. A single build-wide hash would collapse all three into "rebuild
everything", and that is how ingestion cost stops being paid once.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from indexer.core.ids import BuildId, ContentHash, DocumentId, UnitId

__all__ = ["ChangeKind", "DocumentRecord", "Ledger", "PlannedChange", "diff_units"]


class ChangeKind(StrEnum):
    ADDED = "added"
    CHANGED = "changed"
    RESTAGED = "restaged"
    UNCHANGED = "unchanged"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """What the last successful build knows about one document."""

    document_id: DocumentId
    source_uri: str
    content_hash: ContentHash
    unit_ids: tuple[UnitId, ...]
    #: ``StageFingerprint.key()`` per stage, plus one entry per enricher and
    #: per index, so a change can be attributed to the stage that caused it.
    stage_keys: Mapping[str, str]
    build_id: BuildId
    updated_at: str = ""
    #: Non-fatal problems seen while processing, kept so a corpus-wide quality
    #: report does not require a rebuild to produce.
    warnings: tuple[str, ...] = field(default_factory=tuple)
    #: Hash of the scanner metadata the document was built with. The bytes are
    #: not the only input: an ACL changed in a sidecar, or a tenant rule changed
    #: in config, changes what every index must hold for the document while its
    #: content hash stays the same -- and the document was then skipped as
    #: unchanged, keeping the old ACL in every index.
    metadata_hash: str = ""


@dataclass(frozen=True, slots=True)
class PlannedChange:
    """One document's work for this build, decided before any work is done.

    Producing the full plan up front, rather than deciding per document as the
    build streams, is what lets a build report "10 of 510 documents, est. $0.42"
    before spending anything -- and lets a surprising number be questioned
    before rather than after.
    """

    document_id: DocumentId
    kind: ChangeKind
    #: Stages that must re-run. Empty for UNCHANGED and REMOVED.
    stages: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    prior: DocumentRecord | None = None


class Ledger(Protocol):
    """Durable per-document build state. Lives beside the index, not in it.

    Deliberately separate from the indexes: rebuilding a single index (swapping
    embedding models, say) must not lose the knowledge of what has been parsed
    and enriched, which is the expensive part.
    """

    def get(self, document_id: DocumentId) -> DocumentRecord | None: ...

    def put(self, record: DocumentRecord) -> None: ...

    def delete(self, document_id: DocumentId) -> None: ...

    def iter_records(self) -> Iterator[DocumentRecord]: ...

    def document_ids(self) -> frozenset[DocumentId]: ...

    def begin_build(self, build_id: BuildId) -> None:
        """Mark a build in flight, so an interrupted run is resumable.

        Resumability is a stated property: a build that dies after 400 of 500
        documents must resume at 401. Records are committed per document, and
        the in-flight marker distinguishes "not yet processed" from "processed
        and unchanged" on the next run.
        """
        ...

    def commit_build(self, build_id: BuildId) -> None: ...


def diff_units(
    previous: Sequence[UnitId], current: Sequence[UnitId]
) -> tuple[tuple[UnitId, ...], tuple[UnitId, ...]]:
    """Return ``(to_upsert, to_delete)`` for one document's units.

    Unit ids are content-derived, so a unit whose text is unchanged keeps its id
    and needs no re-embedding even when its neighbours moved. This is the payoff
    of keeping position out of ``make_unit_id``.
    """
    prev, cur = set(previous), set(current)
    return tuple(u for u in current if u not in prev), tuple(u for u in previous if u not in cur)

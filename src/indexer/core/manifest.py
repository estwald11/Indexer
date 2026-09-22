"""The build manifest: every version and setting used to build the index.

The manifest answers one question that comes up constantly and is otherwise
unanswerable after the fact: *what produced this index?* Six months later,
retrieval quality has drifted and nobody can say whether the embedding model,
the chunker or the contextualisation prompt changed. Without a manifest the only
honest answer is a bisect over the corpus.

It is also the substrate for invariant 6. An ablation's "before" and "after" are
two manifests plus two eval reports; the delta table is a join over them.

A manifest is written on every build, including incremental ones, and carries
``parent_build_id`` so the chain back to the last full build is walkable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from indexer.core.accounting import StageRun
from indexer.core.ids import BuildId, ContentHash, hash_obj, merge_hashes

__all__ = ["BuildManifest", "CorpusStats", "IndexStats", "new_build_id"]


def new_build_id(config_hash: ContentHash, started_at: str) -> BuildId:
    """Deterministic in its inputs, unique in practice via the timestamp."""
    return BuildId(
        f"build-{started_at.replace(':', '').replace('-', '')}-"
        f"{merge_hashes(config_hash, started_at)[7:15]}"
    )


@dataclass(slots=True)
class CorpusStats:
    documents_total: int = 0
    documents_added: int = 0
    documents_changed: int = 0
    documents_restaged: int = 0
    documents_unchanged: int = 0
    documents_removed: int = 0
    documents_failed: int = 0
    units_total: int = 0
    units_written: int = 0
    units_deleted: int = 0
    units_reused_from_cache: int = 0
    #: Cache entries deleted because no current document uses them any more --
    #: the parse, segmentation and enrichments of removed documents and of the
    #: previous version of edited ones. Counted because erasure that is not
    #: observable is not auditable.
    cache_entries_purged: int = 0
    bytes_parsed: int = 0
    #: Mean over documents. A drop here is an early warning that a corpus has
    #: acquired scans or multi-column layouts the current parser cannot handle.
    mean_reading_order_confidence: float = 1.0


@dataclass(slots=True)
class IndexStats:
    name: str
    kind: str
    impl: str
    unit_count: int = 0
    size_bytes: int | None = None
    #: Dimensions for dense, vocabulary size for lexical, columns for
    #: structured. Kind-specific and uninterpreted by the frame.
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BuildManifest:
    """The record of one build. Written to ``<store>/manifests/<build_id>.json``.

    ``config`` is the *resolved* config -- after defaults, overlays and env
    interpolation -- not the file as written, because the file does not fully
    determine the build and the resolved form does. Secrets are redacted by the
    loader before they ever reach here.
    """

    build_id: BuildId
    config_hash: ContentHash
    library_version: str
    started_at: str
    finished_at: str = ""
    parent_build_id: BuildId | None = None
    incremental: bool = False
    project: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    #: ``StageFingerprint.as_dict()`` per stage instance, keyed by the stage
    #: path in the config (``pipeline.enrich.enrichers[contextualizer]``), so a
    #: fingerprint change points at the line of config that caused it.
    stage_fingerprints: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    corpus: CorpusStats = field(default_factory=CorpusStats)
    indexes: Sequence[IndexStats] = field(default_factory=list)
    #: Aggregated from ``Accountant.by_stage()``.
    stage_totals: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    #: Summed across measured stage runs.
    total_wall_ms: float = 0.0
    #: Wall clock for the whole build, including work outside any stage: the
    #: corpus scan, store writes, ledger commits.
    elapsed_wall_ms: float = 0.0
    #: Stages configured off for this build, with the reason from config. An
    #: ablation manifest is identified by this field alone.
    disabled_stages: Mapping[str, str] = field(default_factory=dict)
    #: Interpreter and installed versions of the packages backing each impl.
    #: An embedding model upgrade that changes nothing else still changes this.
    environment: Mapping[str, str] = field(default_factory=dict)
    #: Non-fatal problems, capped; the full set lives in the ledger per document.
    warnings: Sequence[str] = field(default_factory=list)
    errors: Sequence[str] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        *,
        config: Mapping[str, Any],
        config_hash: ContentHash,
        library_version: str,
        project: str = "",
        parent_build_id: BuildId | None = None,
    ) -> BuildManifest:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        return cls(
            build_id=new_build_id(config_hash, started),
            config_hash=config_hash,
            library_version=library_version,
            started_at=started,
            parent_build_id=parent_build_id,
            incremental=parent_build_id is not None,
            project=project,
            config=dict(config),
        )

    def absorb(self, runs: Sequence[StageRun]) -> None:
        """Fold accounting into the manifest totals."""
        self.total_cost_usd += sum(r.cost_usd for r in runs)
        self.total_wall_ms += sum(r.wall_ms for r in runs)
        for r in runs:
            if r.error:
                self.errors = [*self.errors, f"{r.fingerprint.key()}: {r.error}"]

    def finish(self, elapsed_wall_ms: float = 0.0) -> BuildManifest:
        self.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.elapsed_wall_ms = elapsed_wall_ms
        return self

    @property
    def unaccounted_wall_ms(self) -> float:
        """Build time not attributable to any stage.

        Per-stage accounting is only as useful as its coverage. A build whose
        stages sum to 21s inside a 73s wall clock is not telling you where the
        time goes -- and that was this pipeline, until the gap was printed and
        turned out to be the unit store rewriting a 9MB file once per document.
        Reporting the residual is what makes an unmeasured hot spot findable
        instead of invisible.
        """
        return max(0.0, self.elapsed_wall_ms - self.total_wall_ms)

    @property
    def accounted_fraction(self) -> float:
        return self.total_wall_ms / self.elapsed_wall_ms if self.elapsed_wall_ms else 1.0

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True, default=str)

    @property
    def content_hash(self) -> ContentHash:
        """Identity of this build's *inputs*, ignoring timings and totals.

        Two builds with the same content hash should produce the same index.
        That is the property a reproducibility test asserts, and the reason
        timings are excluded -- they differ on every run and would make the
        hash useless for the comparison it exists to serve.
        """
        return hash_obj(
            {
                "config_hash": self.config_hash,
                "library_version": self.library_version,
                "stage_fingerprints": dict(self.stage_fingerprints),
                "disabled_stages": dict(self.disabled_stages),
                "environment": dict(self.environment),
            }
        )

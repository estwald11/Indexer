"""Exception types.

Kept few and specific. The distinction that matters operationally is between
errors that should stop a build (a bad config, a broken contract) and errors
that should be recorded against one document and moved past (a corrupt PDF in a
corpus of 10,000). A build that halts on the first unreadable file is unusable
at corpus scale; a build that silently drops 8% of documents is worse. So
``DocumentError`` is recorded in the ledger and the manifest, counted in
``documents_failed``, and surfaced in the eval report.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "ContractViolation",
    "DocumentError",
    "IndexerError",
    "StageError",
]


class IndexerError(Exception):
    """Base for everything this library raises."""


class ConfigError(IndexerError):
    """A config file is invalid, or names something that does not exist.

    Always raised before any work begins. A config error discovered mid-build is
    a bug in the loader: validation is total and up front, so that a four-hour
    ingestion never dies at hour three on a typo.
    """


class ContractViolation(IndexerError):
    """An implementation broke its stage contract.

    Distinct from a bug in the implementation's own logic: this is the frame
    catching a violation of something it promised downstream stages, such as a
    parser returning a block whose span does not match its text. Raised loudly,
    because every downstream guarantee -- provenance above all -- depends on it.
    """


class StageError(IndexerError):
    """An implementation failed. Carries the stage for attribution."""

    def __init__(self, stage: str, impl: str, message: str) -> None:
        super().__init__(f"{stage}/{impl}: {message}")
        self.stage = stage
        self.impl = impl


class DocumentError(IndexerError):
    """One document could not be processed. Recorded, counted, and skipped."""

    def __init__(self, document_id: str, stage: str, message: str) -> None:
        super().__init__(f"{document_id} failed at {stage}: {message}")
        self.document_id = document_id
        self.stage = stage

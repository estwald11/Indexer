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

from collections.abc import Sequence
from typing import Any

__all__ = [
    "AccessDenied",
    "ConfigError",
    "ContractViolation",
    "DocumentError",
    "EnrichmentIncomplete",
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


class AccessDenied(IndexerError):
    """A query cannot be answered under the access policy.

    Raised, not degraded: with access control on, a query that states no
    principals has asked a question whose answer depends on who is asking, and
    guessing -- "everyone" or "no one" -- is a leak or an outage that nothing
    in the result would explain.
    """


class DocumentError(IndexerError):
    """One document could not be processed. Recorded, counted, and skipped."""

    def __init__(self, document_id: str, stage: str, message: str) -> None:
        super().__init__(f"{document_id} failed at {stage}: {message}")
        self.document_id = document_id
        self.stage = stage


class EnrichmentIncomplete(IndexerError):
    """Some units of a batch could not be enriched; the others could.

    Raised by an enricher instead of failing the whole batch for one refused or
    truncated answer. ``results`` holds the enrichments that were produced, in
    input order, with ``None`` where one was not; what happens to those units is
    the ``enrich.on_error`` policy, not the enricher's call. The tokens and cost
    of the failed calls travel with it, because they were spent all the same.
    """

    def __init__(
        self,
        results: Sequence[Any],
        errors: Sequence[str],
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        missing = sum(r is None for r in results)
        super().__init__(
            f"{missing} of {len(results)} units not enriched: " + "; ".join(errors[:3])
        )
        self.results = list(results)
        self.errors = list(errors)
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = cost_usd

"""The evaluation harness. Built first, because everything else is a claim.

Import order reflects dependency order: the golden-set format defines what a
correct answer is, metrics define how it is scored, checks define what a valid
pipeline is, and the harness runs them.
"""

from indexer.eval.checks import (
    check_index_surface,
    check_parsed_document,
    check_ranked_list,
    check_unit_stability,
    check_units,
)
from indexer.eval.golden import (
    GoldenQuery,
    GoldenSet,
    GoldOrigin,
    RelevantSpan,
    load_golden_set,
    write_golden_set,
)
from indexer.eval.harness import (
    AblationResult,
    AblationRunner,
    Bootstrapper,
    EvalRunner,
    QueryEngineLike,
    SanityVerdict,
    requires_rebuild,
)
from indexer.eval.metrics import (
    Judge,
    Matcher,
    QueryScore,
    RunReport,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_failed,
)

__all__ = [
    "AblationResult",
    "AblationRunner",
    "Bootstrapper",
    "EvalRunner",
    "GoldOrigin",
    "GoldenQuery",
    "GoldenSet",
    "Judge",
    "Matcher",
    "QueryEngineLike",
    "QueryScore",
    "RelevantSpan",
    "RunReport",
    "SanityVerdict",
    "check_index_surface",
    "check_parsed_document",
    "check_ranked_list",
    "check_unit_stability",
    "check_units",
    "load_golden_set",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "requires_rebuild",
    "retrieval_failed",
    "write_golden_set",
]

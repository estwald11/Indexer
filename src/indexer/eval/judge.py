"""Judges for end-to-end correctness.

The split that matters: **exact comparison for typed answers, a model only for
prose.** A number, a date or a boolean has a right answer that string or numeric
comparison settles exactly, and asking a model instead is slower, costlier and
strictly less reliable. Reserve the model for the cases where meaning actually
has to be read.

A judge is fingerprinted because changing it changes the number. A correctness
score whose judge is unrecorded is not comparable to any other correctness score,
including last week's.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from indexer.core.registry import register
from indexer.core.results import Hit
from indexer.eval.golden import GoldenQuery
from indexer.plugin import StageImpl, dataclass_params
from indexer.textutil import tokenize

__all__ = ["ContainmentJudge", "ExactJudge"]


@dataclass(frozen=True, slots=True)
class ExactParams:
    number_tolerance: float = 1e-6
    #: Types settled by comparison rather than by reading.
    exact_types: list[str] | None = None


@register(
    "judge",
    "exact",
    version="1",
    params_model=dataclass_params(ExactParams),
    summary="Exact comparison for numbers, dates and booleans. No model.",
)
def _make_exact(params: dict[str, Any], **_: Any) -> ExactJudge:
    return ExactJudge(params)


class ExactJudge(StageImpl):
    STAGE, IMPL, VERSION = "judge", "exact", "1"

    def judge(self, item: GoldenQuery, answer: str, hits: Sequence[Hit]) -> bool | None:
        if item.answer is None:
            return None  # nothing to compare against; abstain rather than fail it
        expected = str(item.answer).strip()
        if item.answer_type in ("number", "float", "int"):
            want = _as_float(expected)
            tol = float(self.param("number_tolerance", 1e-6))
            return want is not None and any(
                abs(v - want) <= max(tol, abs(want) * 1e-9) for v in _numbers(answer)
            )
        if item.answer_type == "date":
            return _as_date(expected) in _dates(answer)
        if item.answer_type == "boolean":
            return _as_bool(expected) == _as_bool(answer)
        return expected.casefold() in answer.casefold()


@dataclass(frozen=True, slots=True)
class ContainmentParams:
    #: Fraction of the gold snippet's content words the answer must contain.
    min_coverage: float = 0.6
    fall_back_to_exact: bool = True


@register(
    "judge",
    "containment",
    version="3",
    params_model=dataclass_params(ContainmentParams),
    summary=(
        "Prose: does the answer contain enough of the gold passage's content words. "
        "A stand-in for an LLM judge; strict about wording, so it under-reports."
    ),
)
def _make_containment(params: dict[str, Any], **_: Any) -> ContainmentJudge:
    return ContainmentJudge(params)


class ContainmentJudge(StageImpl):
    """Lexical containment against the gold passage.

    What it actually measures: whether the retrieved evidence contains the gold
    passage's substance. It cannot tell a correct paraphrase from a wrong one,
    so it **under-reports** against a real judge, and systematically -- which
    makes it usable for comparing arms and unusable as an absolute number. The
    report says so rather than presenting it as accuracy.
    """

    STAGE, IMPL, VERSION = "judge", "containment", "3"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._exact = ExactJudge({})

    def judge(self, item: GoldenQuery, answer: str, hits: Sequence[Hit]) -> bool | None:
        if item.answer is not None and self.param("fall_back_to_exact", True):
            return self._exact.judge(item, answer, hits)
        gold_text = " ".join(r.snippet for r in item.relevant if r.snippet)
        if not gold_text.strip():
            # No expected answer and no gold passage -- typically a structured
            # item, whose answer is an aggregate rather than a quotable span.
            # Abstain; scoring it wrong would penalise every arm identically for
            # the golden set's shape rather than for anything they did.
            return None
        gold_terms = {t for t in tokenize(gold_text) if len(t) > 3}
        if not gold_terms:
            return None
        answer_terms = set(tokenize(answer))
        coverage = len(gold_terms & answer_terms) / len(gold_terms)
        return coverage >= float(self.param("min_coverage", 0.6))


_NUM = re.compile(r"-?\d[\d,_]*(?:\.\d+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _as_float(s: str) -> float | None:
    try:
        return float(re.sub(r"[,_]", "", s))
    except ValueError:
        return None


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text):
        v = _as_float(m.group(0))
        if v is not None:
            out.append(v)
    return out


def _as_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _dates(text: str) -> set[date | None]:
    return {_as_date(m.group(0)) for m in _DATE.finditer(text)}


def _as_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "yes", "1", "y")

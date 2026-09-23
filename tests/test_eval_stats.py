"""Evaluation that can be trusted with a decision: splits that do not leak,
differences with an interval, tuning scored on queries it never saw, the agent's
view scored as well as the ranking, and Italian questions for Italian archives."""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from indexer.core.ids import DocumentId
from indexer.core.predicate import Compare, Op
from indexer.core.provenance import Span
from indexer.core.query import Query, QueryType, RoutePath
from indexer.eval.agent_metrics import evaluate_tools
from indexer.eval.bootstrap import HeuristicBootstrapper
from indexer.eval.golden import GoldenQuery, GoldenSet, RelevantSpan
from indexer.eval.harness import AblationResult
from indexer.eval.metrics import QueryScore, RunReport
from indexer.eval.runner import EvalRunner
from indexer.eval.stats import compare, mcnemar, paired_bootstrap
from indexer.eval.tune import tune_fusion
from indexer.impls.route import RulesRouter


def _q(i: int) -> GoldenQuery:
    return GoldenQuery(id=f"q{i:03d}", query=f"question {i}", relevant=())


class TestSplit:
    def test_dev_and_test_partition_the_set(self) -> None:
        gs = GoldenSet(queries=tuple(_q(i) for i in range(200)))
        dev, test = gs.split(0.3)
        assert {q.id for q in dev} | {q.id for q in test} == {q.id for q in gs}
        assert not {q.id for q in dev} & {q.id for q in test}
        assert 40 <= len(dev) <= 80

    def test_adding_queries_never_moves_one_across(self) -> None:
        small = GoldenSet(queries=tuple(_q(i) for i in range(100)))
        large = GoldenSet(queries=tuple(_q(i) for i in range(300)))
        dev_small = {q.id for q in small.split(0.3)[0]}
        dev_large = {q.id for q in large.split(0.3)[0]}
        assert dev_small == {i for i in dev_large if int(i[1:]) < 100}

    def test_a_salt_draws_another_split(self) -> None:
        gs = GoldenSet(queries=tuple(_q(i) for i in range(100)))
        assert {q.id for q in gs.split(0.5)[0]} != {q.id for q in gs.split(0.5, salt="b")[0]}


class TestTests:
    def test_mcnemar_is_exact_for_few_discordant_pairs(self) -> None:
        # 8 improvements, 1 regression: two-sided binomial p = 2 * P(X <= 1) = 20/512.
        base = [False] * 8 + [True] + [True] * 20
        treat = [True] * 8 + [False] + [True] * 20
        b, c, p = mcnemar(base, treat)
        assert (b, c) == (1, 8) and p == pytest.approx(20 / 512)

    def test_mcnemar_sees_no_evidence_in_agreement(self) -> None:
        assert mcnemar([True, False], [True, False]) == (0, 0, 1.0)

    def test_the_bootstrap_interval_brackets_the_mean_and_is_repeatable(self) -> None:
        rng = random.Random(1)
        base = [rng.random() for _ in range(80)]
        treat = [b + 0.1 + rng.gauss(0, 0.05) for b in base]
        delta, lo, hi = paired_bootstrap(base, treat, seed=7)
        assert lo < delta < hi and lo > 0
        assert paired_bootstrap(base, treat, seed=7) == (delta, lo, hi)

    def test_a_noisy_difference_is_not_significant(self) -> None:
        rng = random.Random(2)
        base = [rng.random() for _ in range(30)]
        treat = [rng.random() for _ in range(30)]
        _, lo, hi = paired_bootstrap(base, treat)
        assert lo < 0 < hi


def _report(arm: str, fails: list[bool], ndcg: list[float]) -> RunReport:
    scores = [
        QueryScore(query_id=f"q{i}", failed=f, ndcg={10: n}, recall={20: n}, mrr=n)
        for i, (f, n) in enumerate(zip(fails, ndcg, strict=True))
    ]
    return RunReport(
        arm=arm,
        n_queries=len(scores),
        scores=scores,
        retrieval_failure_rate=sum(fails) / len(fails),
        ndcg={10: sum(ndcg) / len(ndcg)},
    )


def test_the_delta_table_says_which_differences_are_real() -> None:
    n = 40
    base = _report("base", [True] * 12 + [False] * 28, [0.4] * n)
    better = _report("better", [True] * 2 + [False] * 38, [0.6] * n)
    result = AblationResult(reports=[base, better], baseline_arm="base")
    comparisons = {c.metric: c for c in result.significance()["better"]}
    assert comparisons["found@20"].only_treatment == 10 and comparisons["found@20"].significant
    assert comparisons["nDCG@10"].delta == pytest.approx(0.2) and comparisons["nDCG@10"].significant
    table = result.delta_table()
    assert "paired against base" in table and "found@20" in table


def test_errored_queries_are_left_out_of_a_comparison() -> None:
    base = _report("a", [False] * 4, [0.5] * 4)
    treat = _report("b", [False] * 4, [0.5] * 4)
    treat.scores[0].error = "boom"
    assert {c.n for c in compare(base, treat)} == {3}


# --------------------------------------------------------------------------- #
# tuning and agent metrics over a real, small archive                          #
# --------------------------------------------------------------------------- #

CONFIG = """
schema_version: 1
project: {{name: tune}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params: {{root: {data}, include: ["**/*.md"]}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural, max_tokens: 60, params: {{merge_below_tokens: 0}}}}
  enrich: {{enabled: true, enrichers: [{{impl: section_prefix, scope: unit}}]}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25, params: {{fallback_language: it}}}}
      - {{name: dense, kind: dense, impl: hash_embedding, params: {{dim: 64}}}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical, dense], step_budget: 1}}
      iterative: {{targets: [lexical, dense], step_budget: 3}}
"""

TOPICS = {
    "ferie": "Le ferie si richiedono con quindici giorni di anticipo al responsabile.",
    "rimborsi": "I rimborsi chilometrici si liquidano con la busta paga del mese seguente.",
    "sicurezza": "Il corso sulla sicurezza si ripete ogni cinque anni per tutto il personale.",
    "fornitori": "Un fornitore nuovo va qualificato prima del primo ordine.",
    "privacy": "I dati dei clienti si conservano per dieci anni dalla fine del contratto.",
    "orari": "L'orario di ufficio va dalle otto e trenta alle diciassette e trenta.",
}


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Any, GoldenSet]:
    from indexer.pipeline import assemble

    data = tmp_path / "data"
    data.mkdir()
    for name, body in TOPICS.items():
        (data / f"{name}.md").write_text(f"# {name.title()}\n\n{body}\n", encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG.format(root=tmp_path.as_posix(), data=data.as_posix()), encoding="utf-8")
    a = assemble(cfg)
    assert a.ingestion().build().ok
    queries = []
    for uid in a.unit_store.all_ids():
        eu = a.unit_store.get(uid)
        name = eu.unit.metadata["name"].removesuffix(".md")
        span = eu.unit.provenance.span
        queries.append(
            GoldenQuery(
                id=f"g-{name}",
                query=f"regola {name} " + " ".join(eu.unit.text.split()[1:3]),
                relevant=(RelevantSpan(DocumentId(eu.document_id), Span(span.start, span.end)),),
            )
        )
    return a, GoldenSet(queries=tuple(queries))


def test_fusion_weights_are_fitted_on_dev_and_reported_on_test(archive: Any) -> None:
    a, golden = archive
    engine = a.query_engine()
    fuser = engine.fuser
    dev, test = golden.split(0.5)
    result = tune_fusion(
        engine, EvalRunner(), dev, test, indexes=["lexical", "dense"], grid=(0.5, 1.0, 2.0)
    )
    assert engine.fuser is fuser  # restored
    assert len(result.trials) == 3 and result.weights["lexical"] == 1.0
    assert result.dev_tuned >= result.dev_default
    assert "weights: {lexical: 1.0, dense:" in result.render()


def test_the_agent_view_is_scored(archive: Any) -> None:
    from indexer.agent import AgentTools

    a, golden = archive
    report = evaluate_tools(AgentTools(a), golden, top_k=3)
    assert report.n_queries == len(golden) and report.errors == 0
    assert report.found_rate is not None and report.found_rate > 0.5
    assert report.citation_validity == 1.0
    assert report.payload_chars_p50 > 0
    assert "citations valid 100.0%" in report.render()


class TestItalianTemplates:
    def _router(self) -> RulesRouter:
        return RulesRouter(
            {
                "field_lexicon": ["importo", "data_fattura"],
                "field_types": {"importo": "float", "data_fattura": "date"},
                "locale": "it",
                "paths": {"structured": {"targets": ["fields"]}, "lookup": {"targets": ["x"]}},
            }
        )

    def test_structured_questions_are_asked_in_italian_and_read_back_exactly(self) -> None:
        boot = HeuristicBootstrapper({"language": "it"})
        rng = random.Random(3)
        amounts = boot._field_questions(
            "importo", "importo", [100.0, 250.5, 1250.0, 9000.0], 3, rng
        )
        dates = boot._field_questions(
            "data_fattura",
            "data fattura",
            [date(2024, 1, 5), date(2024, 6, 30), date(2025, 2, 1)],
            3,
            rng,
        )
        router = self._router()
        for question, _, qtype in [*amounts, *dates]:
            assert question.startswith("quali documenti hanno")
            d = router.route(Query(text=question), None)  # type: ignore[arg-type]
            assert str(d.path) == RoutePath.STRUCTURED, question
            assert d.structured_query is not None
            clause = d.structured_query.where
            assert isinstance(clause, Compare)
            assert clause.op in (Op.GT, Op.LT)
            if qtype is QueryType.NUMERIC:
                threshold = question.rsplit(" ", 1)[1].replace(".", "").replace(",", ".")
                assert clause.value == pytest.approx(float(threshold))

    def test_english_sets_are_unchanged(self) -> None:
        out = HeuristicBootstrapper({})._field_questions(
            "version", "version", ["1.0", "1.0", "2.0"], 1, random.Random(0)
        )
        assert ("which entries have version 1.0", "1.0", QueryType.STRUCTURED) in out

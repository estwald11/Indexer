# Round 1 review: interfaces and config schema

What to look at, what I decided unilaterally, and the open questions where your
answer changes what gets built.

## What to read, in order

1. `ARCHITECTURE.md` — which invariant forced which contract.
2. `src/indexer/core/stages.py` — the whole frame. Eight protocols, each stating
   input, output, what it may assume, what it must preserve, and its minimal
   implementation.
3. `configs/reference.yaml` — what a project looks like, plus the ablation ladder.
4. `src/indexer/eval/golden.py` — the golden-set format, and the span-anchoring
   decision it rests on.

Everything typechecks under `mypy --strict` and the 41 tests assert the frame's
own guarantees rather than any implementation's.

## The five decisions I would most like argued with

**1. Gold is anchored to `(document_id, span)`, not unit ids.**

The consequence: one golden set survives re-segmentation, so a structural
segmenter and a fixed-window one are comparable. The cost: gold breaks if the
*parser* changes enough to shift the canonical text, and the harness can only
warn (via `RelevantSpan.snippet`) rather than repair. I think this is the right
trade — a golden set that cannot survive a segmenter change cannot measure the
thing most worth measuring — but it is the decision everything else in eval
rests on.

**2. `EnrichedUnit.indexing_text()` is computed once and every index must use it.**

This takes a choice away from index implementations deliberately. An index that
wants to index something else (a title field, raw text in a second column) can,
but the contextualised string must be *a* retrieval surface. The alternative —
letting each index decide — is how the dense and lexical halves drift apart, and
the drift is nearly invisible. If you think an index should be able to opt out,
say so now; it changes `check_index_surface`.

**3. `ContextScope` is declared per enricher and included in the cache key.**

This makes the blast radius of a document edit a config-visible property. The
cost is a contract an implementation can lie about: an enricher declaring `UNIT`
while reading the document will serve stale results after an edit, and the bug
survives cache clears. I can add a runtime guard (pass a restricted view of
`EnrichContext` matching the declared scope) at the cost of some awkwardness in
the batch API. **Worth it?**

**4. The iterative loop lives inside `Retriever`, not as a ninth stage.**

Keeps one pipeline shape for all three paths. The cost is that `Retriever` is
now the most complex protocol, and "retrieve" covers both "call three indexes
once" and "run a four-step agentic loop". If you would rather see a separate
`Planner` stage that the simple path passes through trivially, now is the moment.

**5. `IndexKind` is an open string with no dispatch anywhere.**

This is what makes "a fourth kind touches nothing" true, and it costs static
exhaustiveness: nothing will tell you that a new kind has no router support.
The mitigation is config validation (`Config._coherent` checks that route
targets exist), not the type system.

## Smaller things I decided without asking

- **Pydantic for config, stdlib dataclasses for contracts.** `indexer.core`
  imports nothing third-party, so adopting the frame inherits no dependency.
- **`enabled: false` on every stage, with each stage's pass-through defined in
  its contract.** For `parse` and `segment`, where identity is meaningless, the
  disabled behaviour is the degenerate implementation (decode bytes; one unit
  per document) — both legitimate ablation baselines.
- **Lists replace on overlay, mappings merge.** Otherwise "run without the
  contextualiser" is inexpressible in an ablation arm.
- **Secrets are `${env:VAR}` only, redacted before hashing.** A config file must
  be committable, and rotating a key must not invalidate an index.
- **`on_error: skip` by default for parse and enrich.** A build that halts on
  the first unreadable PDF is unusable at corpus scale; failures are counted in
  the manifest so the corpus cannot silently shrink.
- **The reference path runs entirely offline** (`hash_embedding`, `bm25_memory`,
  `sqlite`, no reranker). It works as a test fixture with no API key, and every
  piece of it is a real ablation baseline rather than a placeholder.

## Decisions (ruled, 2026-09-17)

| Question | Ruling |
|---|---|
| Corpus for the ablation report | No preference — my pick, constrained by the environment (see below). |
| Reference path fidelity | **Offline only.** No model downloads, no credentials. Absolute numbers from the reference dense index are not meaningful; only the deltas are. |
| End-to-end correctness | **Caller-supplied answer function**, defaulting to top-k concatenation. Generation stays out of the library. |
| `ContextScope` enforcement | **Contract only.** No runtime guard. The risk is accepted: an enricher that declares `unit` while reading the document serves stale results through cache clears. `Enricher` docstrings state it; `check_scope_honesty` is *not* being built. |

Not ruled, so built as proposed: the shared retrieval surface stays mandatory
for every index, and the iterative loop stays inside `Retriever`.

## Environment constraints on deliverable 3

Discovered after the rulings, and they change what the ablation report can
prove. Stated here rather than worked around quietly.

The session's network policy allows package registries and `api.anthropic.com`
only. `huggingface.co` and `www.sec.gov` are refused at the proxy (403 on
CONNECT), and no `ANTHROPIC_API_KEY` is set — the API answers 401.

| Wanted | Available | Consequence |
|---|---|---|
| A real corpus | PyPI sdists: README, `docs/**`, changelogs across many packages | Fine. Genuinely heterogeneous (md/rst/txt), real structure, and it carries version, date and requirement facts, so the structured path is exercised rather than stubbed. |
| LLM contextualiser | None | `llm_contextualizer` is written and registered but cannot run here. The ablation measures an **extractive** contextualiser instead. |
| Cross-encoder reranker | None (HF blocked) | `bge_reranker` is written against the contract but cannot run. The ablation measures a **lexical-overlap** reranker instead. |

So the two sanity checks are calibrated for LLM-written summaries and a real
cross-encoder, and this run has neither. The honest expectation is that both
deltas come out **smaller** than the published figures, and a result matching
them exactly would be more suspicious than a result falling short.

What the run can still establish, and what the report will claim: that the
wiring is correct (contextualisation reaches both indexes, reranking reorders
the right candidate set), that the deltas point the right way, and the relative
ordering of the arms. Validating the published magnitudes needs a run with
credentials and model access; the config for it is `configs/full.yaml` and
nothing else has to change.

## What round 2 looks like, assuming this is approved

In this order, because each is testable against the previous:

1. The pipeline orchestrators (ingestion and query) that enforce the contracts —
   including the frame-level invariant-5 checks and the ledger diff.
2. Reference implementations, two per stage: a filesystem scanner; text and
   markdown parsers; structural and fixed-window segmenters; section-prefix and
   LLM contextualisers; BM25, hash-embedding and SQLite indexes; rules and LLM
   routers; sequential and parallel retrievers; RRF and passthrough fusers; noop
   and cross-encoder rerankers.
3. The eval harness implementations: bootstrapper, runner, ablation runner.
4. The ablation report on a real corpus, with the two sanity checks.

I would rather not write (2) before (1) is settled, since the orchestrators are
what reveal whether the contracts are actually sufficient.

## Round 3 (2026-09-18): evidence review

Not a code round. Every invariant and every number quoted in the docs and
docstrings was traced to its primary source and checked against what was
published between the 2024 brief and September 2026. The survey is
`docs/STATE_OF_THE_ART.md`; the changes it produced:

| Where | What changed | Why |
|---|---|---|
| README, `ARCHITECTURE.md` | Invariant 1 re-scoped to its source (a 2026 agent-memory study on LoCoMo); invariant 5 softened from "never" to "never alone"; invariants 3 and 4 given their replication caveats. | The numbers were correct; their scope was not stated. |
| `core/stages.py`, `eval/metrics.py`, `config/schema.py` | Docstrings that quoted the invariant-1 figures as general facts now state where they come from; reranker guidance restated as pointwise-small versus listwise-large. | Same. |
| `configs/reference.yaml` | Arms `hybrid-context-rrf-k20` and `hybrid-context-overlap` added. | RRF `k` and overlap are the two knobs the 2026 literature most often finds mis-set. |
| `configs/full.yaml` | `rerank.input_top_k: 100`; 2026 model names in comments; `sparse` and `tree` index entries sketched, disabled. | Reranking gains flatten near k=100; the new kinds are the seams the evidence opened. |
| `eval/checks.py` | `check_index_surface` probes with the *rarest* context-only term, at a depth of at least its document frequency. | On the PyPI corpus it probed with "Changelog" (2,181 surfaces) and reported both indexes broken when both were correct. |
| `eval/runner.py` | A sanity check that passes within tolerance no longer carries a "larger than expected" diagnosis. | Every PASS in the JSON report read as a warning. |
| `configs/pypi-docs.yaml` | Arm 4 restates `regex_fields` in its enricher override. | Lists replace on overlay, so the arm had silently removed the structured path and measured two changes as one. |
| `tests/test_pipeline.py` | `file://` URI handled with `url2pathname`. | The only test failure on Windows was a test-side path bug. |
| `pyproject.toml` | mypy `python_version` pin removed. | Pinning 3.11 breaks type-checking on 3.12+ hosts where numpy's stubs use `type` statements. |
| `docs/ABLATION-pypi-docs.md` | First ablation on the PyPI-docs corpus. | Deliverable 4 from round 2 had not been recorded. |

What was deliberately *not* changed: no contract, no protocol, no schema field.
The evidence review found the seams in the right places; it did not find a
missing stage.

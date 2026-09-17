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

## Open questions where your answer changes the build

**Q1. What is the real corpus for deliverable 3?**
The ablation report needs one. Without it I will build against a public corpus
(candidates: a slice of arXiv for dense prose, EDGAR filings for tables and
temporal questions, a set of slide decks if you want the visual index exercised
— these stress different halves of the frame). If you have a corpus in mind, its
shape should drive which reference implementations I wire first.

**Q2. Is the offline reference path acceptable as "the simple end-to-end path",
or should it use a real embedding model?**
Offline means the whole thing runs in CI, and the ablation sanity checks are
reproducible on any machine. But `hash_embedding` is not a real dense retriever,
so the *absolute* numbers from the reference path are meaningless — only the
deltas are informative. I lean offline for the reference path plus one
`sentence-transformers` implementation for real numbers. Tell me if you would
rather the reference path be honest end-to-end from the start.

**Q3. How much should the frame own the answer step?**
Right now: nothing. `RetrievalResponse` carries passages. But end-to-end
correctness is a required metric, which means the harness needs *something* to
judge. My plan is for the harness to take a caller-supplied answer function and
default to a trivial one (concatenate top-k) for retrieval-only evaluation —
keeping generation out of the library while making the metric computable.
Confirm, or say you want a thin generation seam in the frame.

**Q4. SQLite or Postgres for the structured store in the reference path?**
The brief says SQLite by default, Postgres when concurrency demands. I will do
SQLite and put the concurrency note in the docs, unless your first project
already needs Postgres.

**Q5. Do you want a CLI in round 2?**
`indexer build`, `indexer query`, `indexer eval`, `indexer ablate`. Not in the
brief, and it is the obvious thing to skip if you would rather see the reference
implementations land first. It is maybe half a day and it is what makes the
ablation runner usable by someone who is not reading the source.

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

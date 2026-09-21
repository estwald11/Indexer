# Ablation report: PyPI documentation corpus

What this run establishes, what it does not, and what it changed about the
contracts.

Of the six invariants the frame is built on, two reproduce cleanly, one holds
but cannot honestly be claimed from this data, two fail with identified causes,
and one fails for a reason this environment cannot fix. **The failures were more
informative than the passes**, which is the argument for building the harness
before the pipeline. Read the caveats before the table.

## The corpus

686 documentation files from 50 PyPI packages: READMEs, `docs/**`, changelogs.
550 reStructuredText, 104 markdown, 32 plain text; 10.4 MB. Fetched by
`scripts/fetch_corpus.py`, which pins each package's version in
`data/pypi-docs/_corpus.json`, so the corpus is reproducible.

Chosen for what it stresses rather than for convenience: heterogeneous formats
(so per-document parse routing is exercised rather than stubbed), genuine
structure (headings, code blocks, pipe tables, directives), wildly variable
length, and the uneven quality that synthetic corpora never reproduce. It also
carries machine-readable facts — versions, release dates, package names — so
the structured path answers real values.

**686 documents → 10,283 units with zero parse and zero segment contract
violations.** That is `check_parsed_document` and `check_units` passing on every
document: every block's span resolves to its own text, blocks are ascending and
non-overlapping, every unit is addressable, every table keeps its grid. Getting
there took four bug fixes, recorded in `ARCHITECTURE.md`.

## Read this before the table

**No neural model could be downloaded.** `huggingface.co` is denied by this
environment's egress policy and there is no `ANTHROPIC_API_KEY`, so no arm here
runs a bi-encoder. Two dense embedders were measured instead, and the difference
between them turned out to be the most informative row in the table:

* `hash_embedding` — a hashing trick over word and character n-grams. A real
  vector index with real write, delete and filter semantics, and *not* an
  embedding model: its projection is fixed before it has seen a document, so it
  cannot match "how do I stop a request hanging" to "timeout" unless the words
  overlap.
* `svd_embedding` — TF-IDF, then a truncated SVD. A space *fitted* on the
  corpus, which is what makes it a model rather than a trick, and still fully
  offline. Weaker than a bi-encoder, which reads syntax and word order where
  this reads only co-occurrence.

Treat every absolute number below as uninterpretable and every delta as the
finding.

**The contextualiser is extractive, not an LLM.** Same reason. This turns out to
matter more than expected — see the first finding.

**The golden set is lexically constructed, and that caps every dense arm.**
`_distinctive_terms` draws each query's detail terms from the target unit's own
text. Measured: those detail terms occur verbatim in the gold passage **91%** of
the time, the subject term **45%** (it comes from the package name, by design,
so contextualisation has something to fix), **75%** overall, and 44% of queries
are covered completely. `GoldenSet.mean_lexical_overlap` reports it beside every
run. BM25 is therefore scored close to precisely what it does, and few items in
the set *require* matching meaning rather than words. This is not a flaw in the
generator — it was built to test contextualisation, and for that the
construction is right — but it is a ceiling on the dense arms, and it binds a
neural bi-encoder exactly as hard as it binds the two embedders measured here.

**The golden set is machine-generated and unverified.** 345 queries from
`HeuristicBootstrapper`, filtered against a lexical baseline: 1,162 candidates
dropped as too easy (the baseline already ranked the gold first, so they cannot
distinguish two systems) and 48 as unfindable. 300 factual, 28 structured, 13
temporal, 4 numeric. No human has checked them.

The 45 structured items are generated from extracted fields, ask for values the
corpus actually holds, and use comparison thresholds drawn from the interior of
each field's range — an earlier version asked for values at the edges, so the
items failed regardless of the system and measured the corpus instead.

Queries are shaped **subject + detail** — a document-level subject plus terms
distinctive to the unit — because that is the shape contextualisation is
supposed to fix: a chunk that never names its own subject. The subject is taken
from the package name rather than from any heading, so the contextualisation arm
does not win by construction.

## Method

Thirteen arms. Each is an override list against one config, so two arms provably
differ in exactly the keys stated — `AblationSpec.overrides` in
`configs/pypi-docs.yaml` is the whole definition of each row.

Each distinct *ingestion* configuration gets its own index store, keyed by a
fingerprint of the arm's `corpus` and `ingestion` sections. Arms that differ
only on the query side reuse the matching build. Without that, reuse would
depend on the order the arms happened to run in — a correctness bug in the
measurement apparatus, which is the worst place to have one. (It was one, for
the first run; see `ARCHITECTURE.md`.)

Scoring uses span-overlap matching at 0.5: a hit counts when it covers at least
half of a gold span. Gold is anchored to `(document_id, span)` rather than unit
ids, which is what lets the same 345 queries score the structural segmenter and
the fixed-window one — arm 8 changes segmentation, and every unit id with it.

The structured path returns records rather than passages, so its retrieval
metrics are reported as not-applicable rather than as failures, and its
correctness is judged on the answer. Routing is scored separately against the
golden item's declared type.

## Reproducing

```bash
python scripts/fetch_corpus.py                        # ~30s, pins versions
python scripts/run_ablation.py configs/pypi-docs.yaml # build + bootstrap + 10 arms
```

Writes `var/pypi/eval/ablation.{json,txt}`. Deterministic: the bootstrapper is
seeded, fusion breaks ties by unit id, and re-running reproduces every arm's
numbers exactly — which is load-bearing rather than tidy. The one bug here that
nothing else would have caught was found by running identical code twice and
getting different answers.

The numeric sections below are rendered from `ablation.json` by

```bash
python scripts/report_table.py var/pypi/eval/ablation.json
```

rather than transcribed. Three revisions of this report introduced two
transcription errors and one comparison quoted from memory; the generator
removes the first failure mode and refuses to print a comparison whose arms it
cannot find in the artifact.


## Results

13 arms, 345 queries, 148s, $0.00. Every number below is rendered from
`ablation.json` by `scripts/report_table.py`.

| arm | P@5 | R@20 | nDCG@10 | fail@20 | correct | route | p50 |
|---|---|---|---|---|---|---|---|
| `1-lexical-only` | 0.145 | 0.917 | 0.573 | **0.072** | 0.825 | 1.00 | 9ms |
| `2-dense-only` (hash) | 0.054 | 0.473 | 0.209 | 0.458 | 0.433 | 1.00 | 5ms |
| `2b-svd-only` (LSA) | 0.100 | 0.797 | 0.390 | 0.177 | 0.647 | 1.00 | 19ms |
| `3-hybrid-rrf` (hash) | 0.105 | 0.847 | 0.411 | 0.133 | 0.699 | 1.00 | 16ms |
| `3c-hybrid-svd` (LSA) | 0.136 | 0.887 | 0.529 | 0.099 | 0.782 | 1.00 | 29ms |
| `3b-hybrid-rerank` | 0.114 | 0.887 | 0.454 | 0.099 | 0.770 | 1.00 | 18ms |
| `4-hybrid-sectionpath` | 0.102 | 0.820 | 0.393 | 0.157 | 0.678 | 1.00 | 15ms |
| `5-hybrid-context` | 0.087 | 0.790 | 0.329 | 0.183 | 0.613 | 1.00 | 18ms |
| `6-hybrid-context-rerank` | 0.103 | 0.853 | 0.406 | 0.128 | 0.696 | 1.00 | 21ms |
| `9-hybrid-weighted` | 0.105 | 0.867 | 0.401 | 0.116 | 0.721 | 1.00 | 20ms |
| `10-context-lean` | 0.110 | 0.897 | 0.450 | 0.090 | 0.770 | 1.00 | 19ms |
| `7-no-router` | 0.089 | 0.853 | 0.406 | 0.258 | 0.696 | 0.00 | 21ms |
| `8-fixed-window` | 0.105 | 0.777 | 0.405 | 0.194 | 0.718 | 1.00 | 18ms |

Query cost is $0.00 in every arm because nothing on the query path calls a
model. That is the whole reason this run exists rather than a better one.

### Controlled comparisons

Each pair differs in exactly the stage named; the override lists in
`configs/pypi-docs.yaml` are the definition.

| change | fail@20 | |
|---|---|---|
| **routing** on → off | 0.128 → 0.258 | **+102%** |
| **segmentation** structural → fixed windows | 0.128 → 0.194 | +52% |
| **embedder** hash → LSA, dense only | 0.458 → 0.177 | **−61%** |
| **embedder** hash → LSA, fused | 0.133 → 0.099 | −26% |
| **reranking** off → on (with context) | 0.183 → 0.128 | −30% |
| **reranking** off → on (no context) | 0.133 → 0.099 | −26% |
| **context** none → verbose | 0.099 → 0.128 | +29% |
| **context** none → chunk-specific | 0.099 → 0.090 | −9% |
| **fusion weights** equal → dense 0.25 | 0.128 → 0.116 | −9% |
| **hybrid** vs its lexical half | 0.072 → 0.099 | +36% |

### What reproduced, and what did not

| Invariant | On this corpus | |
|---|---|---|
| 1. Retrieval predicts answer quality | r = 0.93 | **circular** — see below |
| 2. Ingestion paid once | 24.8s cold, 0.1s warm, per stage | holds |
| 3. Context cuts failures ~⅓ | −9% chunk-specific, **+29% verbose** | falls short / inverts |
| 4. Hybrid beats either half | 0.099 vs **0.072** lexical alone | does not hold |
| 5. Structured never hits vector search | 100% → 0% failure on that slice | **holds, decisively** |
| 6. Nothing optimised without a number | every row above | holds |

Two of six reproduce cleanly, one holds but cannot honestly be claimed from
this data, two fail with identified causes, and one fails for a reason this
environment cannot fix.

### Invariant 5 is the largest effect in the report

Routing is worth more than any other single stage here, and the aggregate
understates it. Holding everything else constant (`6-hybrid-context-rerank`
against `7-no-router`):

| slice | n | router on | router off |
|---|---|---|---|
| structured | 28 | **0.000** | 1.000 |
| numeric | 4 | **0.000** | 1.000 |
| temporal | 13 | **0.000** | 1.000 |
| factual | 300 | 0.147 | 0.147 |
| **all** | 345 | **0.128** | 0.258 |

**Every structured, numeric and temporal question fails without the router and
none fails with it.** The factual slice does not move at all — 0.147 either way
— and that is the point: routing does nothing for the queries retrieval already
handles, and is total for the ones it cannot. *"Which entries have version major
greater than 3"* has no passage that answers it. The answer is an aggregate over
extracted fields, so a vector index cannot return the right thing at any depth.

This is also why `RunReport.by_query_type` exists. Those questions are 13% of
this set; at a more typical 3% the same catastrophic failure would move the
headline by three points and be invisible.

The slice reached 0.000 only after the router was told each field's declared
type — before that, 13 of 28 structured questions compiled to predicates
comparing a string field against a float, matched nothing, and route accuracy
reported 100% throughout. See the eighth entry below.

### Structure-aware segmentation is worth about as much as reranking

`8-fixed-window` (400-token windows, 64 overlap) against
`6-hybrid-context-rerank`, which differs from it in the segmenter and nothing
else: **0.194 against 0.128**, so fixed windows are 52% worse, or structure buys
a 34% reduction. Recall@20 is 0.777 against 0.853. The cost at query time is
zero — the work is all in `segment`.

Fixed windows also fail `check_unit_stability`: because boundaries are
position-derived, inserting a paragraph early in a document shifts every
subsequent boundary and changes every subsequent unit id, so a one-line edit
re-embeds the whole document. The quality loss and the incremental-rebuild cost
come from the same property.

### Reranking works

`5-hybrid-context` → `6-hybrid-context-rerank`: 0.183 → 0.128, **−30%**, inside
the ±50% band around the published −50%, and the only sanity check that passes.
The same change without context (`3-hybrid-rrf` → `3b-hybrid-rerank`) gives
0.133 → 0.099, −26%.

Worth noting *what* is reranking: `lexical_overlap`, which scores query-term
coverage and proximity with no model at all. A cross-encoder should do better.
That a bag-of-words reranker recovers a quarter to a third of top-20 failures
says the first-stage ranking is leaving obvious wins on the table — BM25 rewards
a passage repeating one query term over one containing all of them, and that is
a large share of what reranking fixes.

### Contextualisation: verbose hurts, chunk-specific helps slightly

The most instructive failure in the report. Holding reranking constant:

| arm | context | fail@20 | vs none |
|---|---|---|---|
| `3b-hybrid-rerank` | none | 0.099 | — |
| `6-hybrid-context-rerank` | document lead + topics + path (27% of surface) | 0.128 | **+29%** |
| `10-context-lean` | subject + section path only | 0.090 | −9% |

Verbose context made retrieval substantially *worse*. The cause is measured, not
guessed: `check_context_specificity` reports the prepended text was **76%
identical (token Jaccard) between adjacent units of the same document** and
occupied **27% of the indexed surface**. A quarter of every unit's retrieval
surface was text its neighbours also carried — which cannot rank one unit above
its sibling, while it does inflate length (BM25 penalises that) and pull every
unit's vector toward the document centroid.

Stripping the duplicated parts recovered it and then some: 0.128 → 0.090. The
ladder without reranking says the same — no context 0.133, heading trail 0.157,
full document context 0.183.

This sharpened invariant 3's contract, recorded in `ARCHITECTURE.md`:

> **Context must be chunk-specific, not document-level.** A summary that
> situates *this chunk* is discriminative. Boilerplate describing the document
> is dilution wearing the same shape.

The chunk-specific arm still reaches only −9% against the published −33%, and
the sanity check correctly flags that. The gap is what an LLM-written summary
supplies and an extractive one cannot: a restatement of what the chunk is about,
in vocabulary the chunk does not itself use. `llm_contextualizer` implements the
same contract and needs an API key.

### Invariant 1 reproduces numerically and must not be claimed

Across the thirteen arms, Precision@5 correlates with end-to-end correctness at
**r = 0.926**. Retrieval failure rate correlates at −0.904, recall@20 at 0.936,
nDCG@10 at 0.954.

**This is not evidence.** The correctness figure comes from `ContainmentJudge`,
which asks whether the concatenated top-5 passages contain 60% of the gold
passage's content words. The "answer" is the retrieved text and the judge is
lexical, so judge and metric read the same evidence — a correlation between them
is arithmetic, not a finding. The four metrics correlating equally well is the
tell, and nDCG@10 edging out P@5 is a second: if P@5 were *specifically*
predictive it would separate, and it does not.

Testing invariant 1 honestly needs a generation step and a judge that reads the
answer rather than the passages. The frame supports it — `EvalRunner.answerer`
takes a caller-supplied function and `Judge` is a protocol — and this
environment has no model for either. The correlation is reported because
omitting a number that looks supportive would be worse than printing it with
its caveat.

### Cost and latency

Ingestion: 686 documents → 10,283 units (10.2 MB of text) in **24.8s cold,
0.1s warm**, on one core, $0.00. Per stage: index 15.4s, enrich 4.0s, parse
1.5s, segment 0.5s — 86% of wall clock inside a measured stage. Query: p50
5–29ms by arm, p95 6–45ms.

The warm number is invariant 2 made concrete: a second build of an unchanged
corpus writes nothing, and the ablation runner depends on it — arms sharing an
ingestion configuration reuse the build and report `0 units written`.

The cold number was 73.2s until the accounting was read rather than printed.
Stages summed to 21s of it, and the residual turned out to be the unit store and
the ledger each rewriting their whole file once per document. Both are now
buffered or journalled — a 2.95× speedup, recorded in `ARCHITECTURE.md`. The
coverage figure is reported beside the total for that reason: per-stage timings
with 29% coverage describe a different program than the one that ran.

## The embedder, measured

`2-dense-only` and `3-hybrid-rrf` were read, when this report was first written,
as evidence that dense retrieval does not work on this corpus. That reading
conflated two claims: *dense retrieval does not help here*, and *this particular
projection does not work*. Splitting the embedder out of the store
(`impls/embed.py`) made them separable — `2b-svd-only` and `3c-hybrid-svd`
differ from their counterparts in the embedder key and nothing else.

| arm | P@5 | R@20 | nDCG@10 | fail@20 | p50 |
|---|---|---|---|---|---|
| `1-lexical-only` | 0.145 | 0.917 | 0.573 | **0.072** | 9ms |
| `2-dense-only` (hash) | 0.054 | 0.473 | 0.209 | 0.458 | 5ms |
| `2b-svd-only` (LSA) | 0.100 | 0.797 | 0.390 | **0.177** | 19ms |
| `3-hybrid-rrf` (hash) | 0.105 | 0.847 | 0.411 | 0.133 | 16ms |
| `3c-hybrid-svd` (LSA) | 0.136 | 0.887 | 0.529 | **0.099** | 29ms |

**Most of the dense arm's failure was the embedder: 45.8% to 17.7%, a 61%
reduction, with the store, the fusion, the golden set and the matcher all
unchanged.** Fused, 13.3% to 9.9%, and nDCG@10 from 0.411 to 0.529 — the
largest single-change improvement any arm in this report has produced. Every
pre-existing arm reproduced its numbers exactly across the refactor, which is
the only reason those rows are comparable at all.

Three things this does *not* establish.

*Invariant 4 still does not reproduce.* The hybrid remains worse than its
lexical half, 9.9% against 7.2%. The gap narrowed from +85% to +36%, which is
consistent with the RRF-weighting mechanism in `ARCHITECTURE.md` — a better
dense list displaces fewer good lexical hits — but "narrower" is not "gone".

*LSA is not a bi-encoder.* It reads co-occurrence, not syntax, and it is weakest
where documents are long and queries are short, which is this corpus. It is the
strongest dense index that runs with no model download, not the strongest one
available.

*The remaining gap may not be the embedder's to close.* Given how the golden set
is built, no query here requires a retriever to match meaning. The honest next
experiment is paraphrased queries, not a bigger model.

**Cost.** One SVD over the corpus at `dim=512`: ~8s for 10,283 units, paid at
flush or at the first search after a write, never per query. Dense-only query
latency rises from 5ms to 21ms, because projecting a query costs a sparse row
against a 512x25,953 component matrix. The fitted components are not persisted;
they are rederived from the surfaces on load, which costs about what parsing
them would have.

**A `dim` sweep, for the record.** Dense-only fail@20 on the same index and
golden set: k=128 0.628, k=256 0.541, k=512 0.414, k=1024 0.330. Still improving
at 1024, where the fit costs 32s rather than 8s; `dim=512` is the default
because it is the knee, not because the curve ends there. Character n-grams were
also tried and *hurt* markedly (k=256: 0.655 with, 0.541 without) — they flood
the vocabulary with near-duplicate dimensions and the truncation spends its rank
on orthography rather than topic. They are off by default for that reason.

These sweep numbers come from a standalone harness over the same index and
golden set rather than from ablation arms, so they are directionally comparable
with the table above but were scored without the router in front of them.

## Paraphrasing the set: what was tried, and what it cost

The ceiling above is a property of the queries, so the next experiment is
queries that ask for the passage instead of quoting it. Four surface transforms
were tried first, because they need no model and would have been free. All four
were applied to the same 333 items and scored dense-only against BM25 and
against LSA at `dim=512`:

| transform | bm25 fail@20 | svd fail@20 | gap | example |
|---|---|---|---|---|
| none | 0.165 | 0.414 | -0.249 | `sqlalchemy collection eager loader` |
| question framing | 0.267 | 0.492 | -0.225 | `how do i sqlalchemy collection eager loader` |
| drop rarest term | 0.532 | 0.706 | -0.174 | `sqlalchemy collection eager` |
| re-inflect terms | 0.646 | 0.784 | -0.138 | `sqlalchemy collections eager loaders` |
| inflect + question | 0.718 | 0.832 | -0.114 | `how do i sqlalchemy collections eager loaders` |

**None of them is a paraphrase.** Each makes both arms worse, and the gap
narrows only because everything degrades; the ordering of the arms never
changes. The reason is structural, and it is worth stating because it rules out
a whole family of ideas: the offline dense arms here are bag-of-words methods
too. A word the corpus does not contain — which is what re-inflection produces,
since neither `tokenize` nor BM25 stems — is exactly as opaque to LSA as to
BM25. Dropping a term removes information from both. Question framing adds
stopwords, which both discard.

A useful rewrite has to substitute vocabulary the corpus *does* contain, in the
sense the passage means. Two sources could supply that:

*A human lexicon.* WordNet would do it, and it is retriever-independent, which
matters: a substitution drawn from a distributional model would favour
distributional retrievers by construction, and the LSA arm would win a
comparison it had helped set up. No WordNet data is reachable here — the `wn`
package is the library, and its data downloads from a host the egress policy
denies.

*A language model.* `LLMBootstrapper` (`eval/bootstrap.py`), registered as
`llm_bootstrap`, which `configs/full.yaml` already named. It rewrites each
heuristic item into the query someone would type before they knew the passage's
vocabulary, records the literal form as `source_query`, and stores the resulting
overlap per item. **It needs an API key and there is none in this environment,
so no run of it appears in this report.**

The ordering inside it is the part worth reviewing rather than the prompt.
Answerability is screened on the *literal* query, before the rewrite, using the
same lexical baseline `HeuristicBootstrapper` uses. Screening after the rewrite
is the trap: a lexical baseline rejects a successfully paraphrased query as
"unfindable", which is precisely the property that made it worth generating, and
the resulting set looks carefully filtered while having discarded everything it
existed to add. A rewrite that comes back still sharing more than `max_overlap`
of the passage's vocabulary is tagged `paraphrase_echoed` rather than silently
kept, for the same reason.

Because the rewriter composes the heuristic generator rather than replacing it,
the two sets share candidates, filters and gold spans exactly. The only
difference between them is the wording of the queries, which is what makes
"paraphrasing the set changed the arms by X" a controlled statement.

## Eight ways the harness lied, and how each was caught

The most transferable result here is not a number. Building the harness before
the pipeline was supposed to make the pipeline measurable; what it actually did
first was reveal that **the measurement apparatus was wrong in seven distinct
ways** — and then the pipeline in an eighth — every one of which produced
output that looked like a result.

That shared shape is the point. None of these threw. None produced an obviously
silly figure. Each would have been reported as a finding.

| What was wrong | What it reported | What exposed it |
|---|---|---|
| An empty structured result scored as success | `1-lexical-only` at 0.075 against a then-true 0.110 (0.072 after the later fixes) | `route_acc` dropping to 0.0 in an arm with no business routing differently |
| `enrich.enabled: false` removed field extraction as well as context | every context comparison confounded with an invariant‑5 regression | the same `route_acc` signal, in six more arms |
| The structured index answered any text query with 50 arbitrary rows | a constant noise list entering RRF at rank‑1 weight | the no‑router arm going to 1.000 |
| The ledger committed documents before the indexes holding them were flushed | 493 documents claimed, none held; the resumed build scored an arm at 0.952 | an arm's numbers changing between two runs of identical code |
| The router parsed ISO dates as integers, ignored `before`/`greater than`, and let `version` shadow `version_major` | well‑formed structured questions matching nothing | the structured slice failing at a rate the router's 100% accuracy could not explain |
| The golden set contained 16 identical queries and comparisons at the edges of the data | a structured slice measuring the corpus rather than the system | reading the generated queries |
| The judge returned False for items it could not assess | every arm understated by the same amount — deltas intact, absolutes meaningless | noticing that structured items had stopped carrying a gold span |
| The router was never told the declared type of each field | 13 of 28 structured questions matched nothing, while route accuracy read 100% | the structured slice failing at a rate perfect routing could not explain |

The eighth is the sharpest of them, because it is the one the invariant's own
metric cannot see. Route accuracy asks *did this question reach the structured
index*, and the answer was yes, every time. What it does not ask is whether the
predicate that arrived there could match anything — and "version 1.0.0" read as
the float `1.0` compiles to a comparison against a numeric column that the
value, a string, was never written to. **Routing to the right index with the
wrong predicate has the same outcome as not routing at all**, and it arrives
through a door the metric guarding invariant 5 does not watch. The fix was to
pass along type information the config had stated all along.

Four of the eight were caught by a **cross-check that had no reason to move**:
route accuracy is not a retrieval metric, and an arm that changes only
contextualisation has no business changing it. That is the argument for
reporting per-slice diagnostics next to the headline rather than instead of it —
not because anyone reads them routinely, but because they are what disagrees
when the headline is wrong.

Two were caught by **reporting a residual**: per-stage accounting that summed to
21s inside a 73s build, and a manifest that now carries the difference. A
coverage number beside a total is cheap and it found two bugs neither total
would have.

One was caught by **re-running identical code and getting different numbers**,
which is the only reason the durability bug was found at all. Determinism is not
a nicety in an evaluation harness; it is the property that makes every other
inconsistency visible.

The general lesson, and the one worth carrying to another project: **an
evaluation harness fails silently by construction.** Its output is numbers, and
wrong numbers look exactly like right ones. Every guard above is cheap, and
none of them would have been added in response to a symptom, because there was
no symptom — there was a plausible table.

## What this changes about the defaults

Findings that are now encoded rather than written down, so the next project
inherits them:

| Finding | Where it lives now |
|---|---|
| Context must be chunk-specific | `check_context_specificity`, run by the ablation script and reported in `ablation.json` |
| Fusion weights are not optional tuning | `Fuser` contract in `ARCHITECTURE.md`; equal weights kept as the prior, not the answer |
| `segment.overlap_tokens` earns nothing | Stays 0 by default; kept only as an ablation arm |
| Accounting needs a coverage number | `BuildManifest.unaccounted_wall_ms`, printed in every build summary |
| Arms must differ only in stated keys | `AblationSpec.overrides`; two confounded comparisons were caught this way |

## What is still open

Ordered by how much each would change the conclusions.

**1. A bi-encoder.** Every dense number here is a floor. `2-dense-only` to
`2b-svd-only` cut failures 61% by changing the embedder alone, and LSA is the
strongest dense index that runs with no model download — not the strongest one
available. Invariant 4 failing on this corpus is a statement about a hashing
trick and a truncated SVD, not about hybrid retrieval.

**2. Paraphrased queries.** The golden set's queries share vocabulary with their
gold passages, so no query here *requires* a retriever to match meaning. That
caps what any dense arm can demonstrate. `LLMBootstrapper` is written and
registered for exactly this and needs an API key; four model-free transforms
were tried and none is a paraphrase (see above).

**3. An LLM contextualiser.** The chunk-specific arm reaches −9% against the
published −33%. The gap is what a written summary supplies and an extractive one
cannot: a restatement of the chunk in vocabulary it does not itself use.
`llm_contextualizer` implements the same contract.

**4. An independent judge.** Correctness here is lexical containment over the
retrieved passages, which is why the r=0.911 correlation with P@5 is arithmetic
rather than evidence. A generation step and a judge that reads the answer would
make invariant 1 testable. `EvalRunner.answerer` and the `Judge` protocol are
the seams.

**5. Human verification of the golden set.** 333 machine-generated items, zero
verified. Every absolute number in this report inherits whatever bias the
generator has.

None of the five needs a frame change. That is the claim the whole exercise was
meant to test, and it is the one result here that came out as hoped.

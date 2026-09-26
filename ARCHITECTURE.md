# Architecture

This document exists for one reader: the maintainer who has new evidence and
needs to know what to revisit. Every contract below is traced to the invariant
that forced it. When an invariant stops being true, this tells you what was
built on it.

Contracts are stated in `src/indexer/core/stages.py` and made executable in
`src/indexer/eval/checks.py`. This document says *why* they are what they are.

---

## The shape

```
                     ingestion (paid once)
   corpus ──▶ parse ──▶ segment ──▶ enrich ──▶ index
                 │          │          │          │
             ParsedDoc    Unit    EnrichedUnit   ...indexes
                                                    │
   query ──▶ route ──▶ retrieve ──▶ fuse ──▶ rerank ┘
                          query (paid forever)
```

Eight stages. Each is a `Protocol` with a contract; each has at least two
implementations selected by config; no stage imports another's internals.

The types crossing each boundary are in `indexer.core` and are dependency-free
by policy. A project that adopts this frame inherits the types and none of the
choices.

---

## Invariant 1 — retrieval is the bottleneck, not generation

> Retrieval failures drive 11–46% of end-to-end errors while utilization
> failures stay at 4–8% regardless of configuration. Precision@5 predicts answer
> accuracy at r=0.98.

**Where those numbers come from, and how far they reach.** Yuan, Su and Yao,
*Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory*
(arXiv 2603.02473, March 2026). The figures are quoted correctly, but the study
is about agent *memory* on one conversational benchmark (LoCoMo), nine
configurations, one reader model; its "retrieval failure" bucket includes
write-side failures, and r=0.98 is a correlation across nine points. The
invariant survives because the same direction shows up everywhere else:
oracle-versus-retrieved gaps of 7–30 points recur across 2025–26 document
benchmarks (T²-RAGBench, Akarsu et al.). The percentages are one study's; the
priority they imply is the field's. See `docs/STATE_OF_THE_ART.md` §0.

**What it forced.**

*The library stops at retrieval.* `RetrievalResponse` carries passages, a route
decision and a trace — not an answer. Generation is the caller's. Spending the
frame's complexity budget on the 4–8% would be spending it in the wrong place,
and a library that owns generation inevitably starts optimising for it.

*Precision@5 is not optional.* `EvalConfig.k_values` is validated to include 5,
and a config omitting it produces a warning. When there is room for one number
on a dashboard, this is the one: cheap to compute, and the metric that tracked
answer accuracy most closely in the study above. nDCG@10 is the one the
2025–26 benchmarks report; both are in the table.

*Retrieval failure rate is reported separately from recall.* `RunReport` carries
both because they answer different questions. Recall@20 averages coverage;
`retrieval_failure_rate` counts the queries with **nothing** relevant in the top
k — the ones that cannot be answered however good generation is. A change that
lifts mean recall while leaving the failure rate flat has improved nothing this
invariant cares about.

**Revisit when:** the utilisation number climbs — long-context models that
genuinely use the middle of their window would move it. The frame would then
need to own more of the answer path, and `RetrievalResponse` is where that
starts.

---

## Invariant 2 — ingestion cost is paid once, query cost forever

**What it forced.**

*Content addressing at every stage boundary.* `cache_key()` is
`H(stage, impl, version, params_hash, input_hash, scope_hash)`. Not a
convenience: it is what makes "once" really once, across runs and machines.

*A hard purity rule.* A stage must be a pure function of its declared inputs.
Not enforceable by types, so it is stated in every protocol docstring and it is
the first thing to suspect when a cache misbehaves.

*Version honesty.* `Registration.version` is a **behavioural** version, not a
package version. Any change that can change output must bump it or change
`params_hash`. An edited prompt without a bump serves stale cache entries
forever, and the symptom — the change appearing to do nothing — reads as
evidence against the change.

*A ledger, separate from the cache.* `indexer.core.ledger`. A cache knows
whether a computation has been done; it cannot know a document was **deleted**
from the corpus, so its units answer queries forever. The ledger holds
per-document state and turns a build into a diff: ADDED / CHANGED / RESTAGED /
UNCHANGED / REMOVED.

*Per-stage fingerprints in that ledger, not one build hash.* This is what makes
RESTAGED possible. Changing the reranker reprocesses nothing; changing the
segmenter reprocesses segment onward; changing one enricher reruns that enricher
only. A single build-wide hash collapses all three into "rebuild everything",
and the invariant is lost.

*Position out of unit identity.* `make_unit_id(document_id, content_hash,
occurrence)` — no ordinal. Inserting a paragraph on page 1 must not change the
id of every unit after it, or a one-line edit re-embeds the document. Tested in
`TestUnitIdentityIsPositionIndependent`.

**Revisit when:** embedding becomes near-free. Much of this machinery is
amortising a cost that a step change in inference price would make negligible —
though `enrich` would still dominate, and the ledger would still be needed for
deletions.

---

## Invariant 3 — chunks must carry their context

> An LLM-written 50–100 token summary prepended before both embedding and
> lexical indexing cuts top-20 retrieval failure from 5.7% to 2.9%; reranking
> takes it to 1.9%.

This is the invariant with the most structural consequence, because of the word
**both**.

**What it forced.**

*`EnrichedUnit.indexing_text()` — one retrieval surface, computed once.* The
failure mode is mundane and near-invisible: the dense index gets the
contextualised string, the lexical index gets the raw one, because two
implementations each decided what to index. Retrieval still works, just worse,
and the ablation shows contextualisation earning about half of what it should.

So implementations do not decide. The string is computed on the unit, and every
`Index` is contractually required to use it as its retrieval surface. It is also
checked: `check_index_surface()` finds a unit whose *context* contains a term
absent from its body, searches for that term, and requires the unit back.

*Contextualisation off degrades cleanly.* With no `context` enrichments,
`indexing_text()` returns `unit.text` and every index follows automatically.
That is the ablation arm, and it costs one config key.

*`enrich` is a chain of independent enrichers, not one step.* Each writes its
own key in `EnrichedUnit.enrichments` and never mutates another's. That is what
makes them individually disableable.

*`ContextScope` is declared per enricher.* It states what the enricher reads,
and the cache key includes exactly that. A `UNIT`-scoped enricher survives edits
elsewhere in its document; a `DOCUMENT`-scoped one — contextualisation, by
nature — does not. This is the honest cost of document-level context, stated in
config rather than discovered during a rebuild. A `RELATED`-scoped one also
reads the documents linked to its own (an attachment's message), and an edit
to any of them restages it (Round 3).

*Enrichers receive batches, not units.* `Enricher.enrich` takes a sequence and
`EnrichContext` carries the whole parent document, because the reference
approach is a small fast model with prompt caching over the parent. That only
pays if units arrive batched by document.

*Overlap defaults to 0.* `SegmentConfig.overlap_tokens` exists only as an
ablation arm. Overlap is a chunking-era workaround for lost context; this
invariant says the fix is contextualisation. Worth a number on a new corpus,
not worth a default.

**What the 2025–26 evidence adds.** The Anthropic figures are vendor-internal
and no independent reproduction of the full 5.7% → 1.9% ladder exists.
Merola & Singh (2025) measure +5.8% nDCG@10 for contextual retrieval on an
NFCorpus subsample, at ~20 GB VRAM; Zhou et al. (*Beyond Chunk-Then-Embed*,
Feb 2026) find contextualisation improves in-corpus retrieval but *degrades
in-document* retrieval. Two alternatives now give a chunk its context without
an LLM summary: late chunking (Jina 2024; ConTEB 2025, +9 nDCG@10 untrained) and
document-batched contextual embedders (Voyage `voyage-context-4`, vendor
figures). Both attach inside the dense `Index` implementation, because the
embedder has to see the whole document, and the enrich stage already delivers
units batched by document. The section-prefix control arm is more important,
not less: the 2026 literature's main complaint about contextualisation papers
is the missing "does any prefix help?" baseline.

**Revisit when:** the numbers move on your corpus. They are published figures,
not laws. `configs/reference.yaml` encodes them as `sanity_checks`, so a corpus
that disagrees says so on the first ablation run. Read a failed check as "the
effect is smaller here", not "the wiring is wrong", until the diagnosis list
has been walked.

**What measuring it added to the contract.** The published framing says "an
LLM-written 50-100 token summary". Running the ablation with an *extractive*
contextualiser made retrieval measurably worse (13.8% → 18.9% top-20 failure),
and the diagnosis sharpens what the invariant actually requires.

Measured on the corpus: the prepended context was **75.6% identical (token
Jaccard) between adjacent units of the same document**, and occupied **27% of
the indexed surface**. So a quarter of every unit's retrieval surface was text
its neighbours also carried. That cannot help distinguish one unit from another
inside a document — which is exactly the discrimination retrieval needs — while
it does inflate length (BM25 penalises it) and pull every unit's dense vector
toward the document centroid.

The requirement the published number leaves implicit, and that this frame should
state:

> **Context must be chunk-specific, not document-level.** A summary that
> situates *this chunk* is discriminative. Boilerplate describing the document
> is dilution wearing the same shape.

`Enricher` contracts now say so, and `ContextScope.DOCUMENT` means "reads the
document", never "writes the same thing for every unit in it". The distinction
is not checkable from a type, so `indexer.eval.checks.check_context_specificity`
measures it and the ablation report prints it.

---

## Invariant 4 — hybrid beats either half

**What it forced.**

*Several indexes over the same units, fanned out over uniformly.* `Index` has
one shape; `Retriever` calls `search` on each target; nothing enumerates kinds.

*`IndexKind` is a plain string and nothing branches on it.* The requirement was
"adding a fourth kind must not require touching the other three". An enum would
have to be widened; a union would need a member; a dispatch table would need a
row. A string with no dispatch needs none of that. `configs/full.yaml` adds a
visual index as four lines of YAML.

*Capabilities instead of a fat interface.* A structured index answers
`StructuredQuery`, which a dense index cannot. Putting that on `Index` would
force every vector store to stub it, and each new capability would widen the
interface every index must satisfy. So `StructuredCapable` is a separate
protocol and the frame asks `isinstance`. A fifth kind with its own capability
adds a protocol and touches no existing index.

*Rank, not score, is the fusion currency.* `Hit.rank` is mandatory and 1-based;
`Hit.score` is documented as index-local and non-comparable. BM25 scores and
cosine similarities are not on a common scale, and a fuser that adds them
asserts a calibration it does not have. `RankedList` rejects sparse ranks at
construction — a gap would silently distort every RRF score.

*Retrieval failures are isolated.* One index erroring degrades the candidate set
and records the error; it does not fail the query. A reranker over three lists
works with two.

**What the 2025–26 evidence adds.** Hybrid still wins with 2026 embedders:
Vespa (Jan 2026) measures +3–5 nDCG points for BM25 + vector over vector alone
on every sub-500M model tested; Akarsu et al. (Apr 2026) find BM25 *beating*
text-embedding-3-large on 23k financial queries and hybrid RRF beating both.
Two caveats are now measured. *Balancing the Blend* (2025): fusion is only as
good as its weakest leg, so a third leg (learned sparse, visual) has to earn its
place on the ablation table rather than being added because it exists.
*Drowning in Documents* (2025) and the 2026 compute-allocation study: reranker
gains grow to about k=100 and then flatten or reverse, so `input_top_k` is a
tuned parameter, not a default. On fusion functions, Bruch, Gai & Ingber (2023)
remain the reference: RRF is sensitive to `k`, and a convex combination of
normalised scores with a fitted weight beats it when a few labelled queries are
available. `k=60` is a prior; `configs/reference.yaml` now carries a `k=20` arm.

**Revisit when:** a single retriever genuinely dominates on your corpus. The
frame does not require hybrid — `dense-only` and `lexical-only` are two of the
arms in `configs/reference.yaml` precisely so this can be checked rather than
assumed.

**What measuring it added to the contract.** On the ablation corpus, hybrid was
*worse* than its lexical half alone: 7.5% top-20 failure for BM25, 47.4% for the
dense index, 13.8% fused with equal RRF weights. Down-weighting dense to 0.25
recovered it to 12.0%, but never past lexical alone.

This does not refute the invariant — the dense arm here is a hashing trick, not
an embedding model, so it contributes noise rather than complementary semantics.
What it shows is the mechanism, and the mechanism generalises: **RRF assumes the
lists it fuses are of comparable quality.** With `k=60` and 50 candidates per
list, an irrelevant document at rank 1 of the weak list scores `1/61 = 0.0164`,
which outranks a relevant document at rank 10 of the strong list (`1/70 =
0.0143`). A bad retriever does not merely fail to help; it actively displaces a
good one's mid-ranked hits.

So the contract note on `Fuser`: **weights are not a tuning nicety, they are how
a hybrid survives one half being worse than the other.** Equal weights are a
claim that both halves are equally trustworthy, and that claim should be checked
before it is made. `configs/*.yaml` ship equal weights because that is the right
*prior*; the ablation is how a corpus corrects it.

**What testing that explanation added.** The paragraph above asserts a cause —
"the dense arm is a hashing trick, not an embedding model" — and an assertion
about why a measurement came out badly is worth no more than the measurement
itself until it is tested. So the embedder was split out of the dense index
(`impls/embed.py`), and `svd_embedding` was added: the same store, the same
fusion, the same golden set, with the fixed projection replaced by one *fitted*
on the corpus. Arms `2b-svd-only` and `3c-hybrid-svd` differ from `2-dense-only`
and `3-hybrid-rrf` in that key and nothing else.

| | dense-only fail@20 | hybrid fail@20 | hybrid nDCG@10 |
|---|---|---|---|
| `hash_embedding` | 47.4% | 13.8% | 0.411 |
| `svd_embedding` | **18.3%** | **10.2%** | **0.529** |
| lexical alone | — | 7.5% | 0.573 |

The explanation held: **61% of the dense arm's failures were the embedder**, and
they came back with no other change. Two things it did not do. It did not make
the hybrid beat its lexical half — 10.2% against 7.5% — so invariant 4 still
does not reproduce here, though the gap narrowed from +84% to +36%. And it did
not make the frame move: every pre-existing arm reproduced its numbers exactly
across the refactor, which is the only reason the two rows above are comparable
at all.

What that leaves is a second explanation, and it is the golden set rather than
the embedder. `HeuristicBootstrapper` draws each query's detail terms from the
target unit's own text (`_distinctive_terms`), and measuring it says how far
that goes: **the detail terms occur verbatim in the gold passage 91% of the
time**, the subject term only 45% (it comes from the package name, deliberately,
so contextualisation has something to fix), and 44% of queries are covered
completely. `GoldenSet.mean_lexical_overlap` now reports this, because a
retrieval number without it cannot be read: a set this lexical scores a word
matcher on exactly what it does, and few of its items *require* matching
meaning. That ceiling binds a neural bi-encoder as hard as it binds LSA. **To
test invariant 4 properly this corpus needs paraphrased queries, not a better
embedder** — which is a bootstrapper change. `LLMBootstrapper` is that
change; what it cost to get right is below.

---

## Invariant 5 — structured, numeric and temporal questions must never reach vector search

**What it forced.**

*`route` as a first-class stage with at least three paths.* `RouteConfig`
validation rejects a config missing `structured`, `lookup` or `iterative`.

*Structural enforcement, not just router good behaviour.* `RouteDecision`
raises if a `STRUCTURED` decision carries no `structured_query` — otherwise it
falls through to vector search and violates the invariant silently. `LOOKUP`
raises if `step_budget != 1`, so "one retrieval pass" is a guarantee.

*Config-level enforcement too.* `Config._coherent` rejects a config whose
structured path targets no structured index. The router cannot honour this
invariant if the configuration makes it impossible, and the failure would be a
silent quality problem rather than an error.

*A predicate AST.* `indexer.core.predicate` — small enough to compile to SQL, to
vector-store filters, and to a pure-Python evaluator (which is the test of
whether it is small enough). Structured questions need somewhere to go, and it
has to be vendor-neutral or the invariant buys a vendor lock.

*`Exists` distinct from `Compare(field, EQ, None)`.* "No governing-law clause
found" and "governing law is explicitly none" are different answers. An
extraction pipeline that cannot tell them apart will confidently report the
wrong one.

*Typed field extraction in `enrich`.* `Enrichment.fields` with a narrow
`FieldValue` union. The structured path can only answer from fields that were
extracted; without extraction there is nowhere for a numeric question to go.

*Per-type metric slices.* `RunReport.by_query_type`. Aggregates hide this
failure mode entirely: structured questions are a minority of most golden sets,
so routing them all to vector search costs a couple of points overall and is
invisible — while being catastrophic for that slice.

*Every decision logged, including the non-decisions.* When routing is disabled
the pipeline still records a decision with `reason="stage_disabled"`. Routing
errors are invisible in aggregate retrieval metrics, because the misrouted
queries are exactly the ones whose gold the retriever never saw — the metrics
blame the retriever.

**What the 2025–26 evidence adds.** The direction is supported; the word
"never" is stronger than the evidence. TableRAG (EMNLP 2025) shows SQL execution
over preserved tables beating flattened-table retrieval on aggregation and
nested questions (HeteQA 44.19% vs 34.54%), and T²-RAGBench (EACL 2026) puts
oracle-context numerical accuracy near 72% against ~41% for the best retrieval
pipeline — the bottleneck is finding the table, not the arithmetic. Text or
hybrid retrieval is still what *locates* it. The frame already behaves this
way: `RulesRouter` falls back to LOOKUP when no predicate is extractable and
records why. What the evidence asks for is a structured *executor* (SQL over
extracted rows, which `StructuredCapable` + `sqlite` already is) rather than a
structured *filter* alone, and a per-type metric slice so misrouting is
visible — which `RunReport.by_query_type` provides. On routers themselves,
RAGRouter-Bench (Apr 2026) finds a TF-IDF + SVM complexity router at macro-F1
0.928 saving 28% of tokens, which vindicates the rules router as the reference.

**Revisit when:** retrieval models start handling numeric and temporal
constraints natively. The seam is `RoutePath`, which is open: a fourth path
costs a config entry.

---

## Invariant 6 — nothing is optimized without a before/after number

**What it forced.**

*The eval harness first, before implementations.* Which is why
`src/indexer/eval` exists in round 1 and `src/indexer/impls` does not.

*Gold anchored to document spans, not unit ids.* The single most consequential
decision in the eval design. Unit ids are content-derived, so changing the
segmenter changes all of them and a unit-id-anchored golden set silently reports
zero recall — which looks like a catastrophic regression rather than a broken
harness. Span anchoring means one golden set survives re-segmentation,
re-chunking and re-parsing, which is what makes ablation arms comparable at all.

*Accounting is mandatory, not opt-in.* Every stage call goes through
`Accountant.measure`. A number that is expensive to obtain is a number nobody
obtains, and the invariant quietly stops being followed.

*Failures are recorded, not just successes.* A stage that fails fast on 30% of
documents is cheap and useless; an accounting layer that only sees successes
reports it as cheap.

*A manifest per build.* Every version and setting, including the resolved
config, the environment, and which stages were disabled. Six months later,
"what produced this index?" has an answer that is not a bisect.

*Ablation arms as override lists, not separate config files.* `AblationSpec
.overrides` are dotted paths. Two arms then provably differ in exactly the
stated keys, and each row of the delta table is attributable to one change.

*List elements addressable by name.* `indexes[lexical].enabled` rather than
`indexes.0.enabled`, because an index's position in a list is not a stable thing
to write into an ablation spec.

*Rebuild vs reuse is derived.* `requires_rebuild()` — arms touching `ingestion`,
`corpus`, `paths` or `cache` rebuild; arms touching only `query` reuse the
index. Otherwise every ablation pays full ingestion cost and nobody runs them.

*The published expectations are asserted.* `SanityCheck` encodes the two from
the brief: contextualisation ≈ −⅓ on retrieval failures, reranking ≈ −½ again.
`SanityVerdict.diagnosis` lists the usual causes in order, because "something is
wired wrong, investigate" is much more useful with a checklist attached.

---

## Decisions not forced by an invariant

Choices made on general grounds. These are the ones to argue with first, because
nothing in the evidence pins them.

**Protocols, not ABCs.** Structural typing means an implementation need not
import the frame to satisfy it — useful for wrapping an existing retriever you
want to benchmark against, which is usually the first thing asked for.

**Frozen dataclasses for data, pydantic for config only.** Config is where
validation earns a dependency; the contract types must stay dependency-free so
adopting the frame imports nothing. `indexer.core` has no third-party imports
and there is a test that will eventually enforce it.

**Bytes in the cache, not objects.** A pickle cache makes every dataclass change
a silent corpus-wide invalidation or an unpickling error.

**`ArtifactStore` separate from `CacheStore`.** Different lifetimes. A cache
entry may be evicted and recomputed; an artifact is referenced by a `MediaRef`
held in an index, and evicting it breaks provenance. Conflating them makes a
cache clear corrupt the corpus.

**Canonical text as the provenance coordinate system.** Byte offsets into a PDF
are meaningless and into HTML point at markup. `ParsedDocument.text` with
`text[span] == block.text` is checkable, which turns provenance from a promise
into a test.

**`reading_order_confidence` on every parse.** So a parser that cannot recover
order declares it rather than emitting stream order and letting the segmenter
build nonsense from it. Also an early warning that a corpus has acquired scans.

**The iterative loop lives inside `Retriever`, not a ninth stage.** A separate
stage would give the agentic path a different pipeline shape from the simple
path, and every downstream stage would need to know which it was in.

**`Hit.matched_text` distinct from `unit.text`.** The retrieval surface includes
LLM-written context. Showing it as if it were the document is a fabricated
citation.

---

## Seams left open

Out of scope, with the place each would attach.

| Deferred | Where it attaches |
|---|---|
| UI | `RetrievalResponse` carries everything a citation view needs: provenance with page and bbox, the full trace, per-stage timings. |
| Agent loop | The tools an agent calls exist (`indexer.agent`, served over MCP); the loop that decides when to call them is the agent's. `Query.context` carries prior turns, and the LLM router turns a follow-up into a standalone question. |
| Vendor coupling | Every vendor sits behind `Registration` and an extra. `indexer.core` imports nothing third-party. |
| Distributed indexing | `Ledger` and `CacheStore` are protocols; a distributed build needs a shared ledger with per-document locking and a shared cache. `PlannedChange` is already a partitionable work list. |
| Quantization | Inside a dense `Index` implementation. `IndexStatsView.detail` carries the knobs; the Matryoshka note in `configs/full.yaml` is there so the two-stage option is not foreclosed. |
| Multi-tenancy | `Predicate` filters push down to every index, `SourceSpec.namespace` scopes document ids, and `query.access` scopes every path -- structured included -- by the caller's principals. What is missing is per-tenant index isolation, which is an `Index` implementation concern. |
| A shared store | The ledger, unit store and SQLite index are files; a Postgres-backed `Ledger`, `CacheStore` and structured `Index` would let several builders and many readers share one archive. Each is a protocol already. |
| Late chunking / contextual embeddings | Inside the dense `Index`: the embedder sees the whole document and returns one vector per unit. `Index.upsert` already receives units batched by document. |
| Visual (page-image) index | The `visual` kind in `configs/full.yaml`. ViDoRe v3 (2026) says: route per page, fuse with the text legs, and rerank with a *text* reranker; store pooled, truncated or binarised multi-vectors. |
| Tree / table-of-contents index | A `tree` kind: TOC tree with node summaries (PageIndex-style, ~$0.001/page), plus a section-as-file layout of `ParsedDocument` so grep-style agents can navigate at zero model cost. Routed to for long single-document analytical questions. |
| Learned sparse leg | A third first-stage index (`SPLADE-v3`, OpenSearch neural-sparse v3-gte) behind the same `Index` protocol. Must earn its place: a weak leg drags fusion down. |
| Graph enrichment | An optional `Enricher` writing entities and relations into `Enrichment.extra`, gated on a multi-hop-heavy query mix. Fair 2026 benchmarks (GraphRAG-Bench, WildGraphBench) show it losing to hybrid + rerank on fact retrieval. |
| Deletion semantics | `Index.delete` says "must actually remove". *Ghost Vectors* (Jun 2026) shows soft-deleted vectors stay recoverable in HNSW files, and most stores compact lazily. The manifest should record the store's compaction policy; a compaction or key-rotation step belongs after any bulk re-index. |
| Embedding-model migration | Drift-Adapter (EMNLP 2025) and shared-space model families (Voyage 4) make "swap the model without re-embedding everything" possible. The gate is the eval harness: 200–500 labelled queries before cutover. |

---

## What implementing it changed

The contracts were proposed before any implementation existed. Building the
reference path found five places where they were wrong or incomplete. They are
recorded because the corrections are the evidence that the contracts are load-
bearing rather than decorative.

**`Unit.verbatim` did not exist.** The segment contract said a unit's text must
be locatable in its span, and permitted a split table to repeat its header rows
into each piece. Those two are contradictory: a repeated header is text that is
not at the span. The checker caught it on the first real corpus. The fix is a
declared flag rather than a tolerance in the checker -- a derived unit says so,
and non-verbatim units are still checked (their text must plausibly come from
their span), so the exception cannot quietly widen to cover real bugs.

**`_split_long_block` returned strings.** Splitting an oversized block at
sentence boundaries and re-joining with `" "` discards the original newlines,
so the unit no longer matched the text it cited. It now returns offsets and the
caller slices. The general lesson, which applies to any parser or segmenter
added later: **derive text from the canonical text, never rebuild it.**

**`Flushable` did not exist.** Indexes persisted on every `upsert`, which is
quadratic for any implementation that rewrites a file or reindexes on commit --
invisible on a ten-document test, fatal at 10,000 units. The capability lets an
index buffer and the pipeline commit once per build. It is a capability rather
than a method on `Index` for the same reason `StructuredCapable` is: an index
with nothing to commit should not have to stub it.

**Ablation arms shared one store.** Arms that "reuse the index" were reading
whatever the previous arm left behind, so reuse depended on the order the arms
ran in. The runner now keys each store by a fingerprint of the arm's `corpus`
and `ingestion` sections, which makes reuse a fact about the configuration
rather than about the loop. This was a correctness bug in the measurement
apparatus, which is the worst place to have one.

**A schema default carried an interpolation token.** `decision_log` defaulted to
`"${paths.store}/route-decisions.jsonl"`, but interpolation runs over the config
file *before* validation, so a default is never expanded -- and the literal
string created a directory named `${paths.store}`. Defaults now resolve in the
assembler. The general rule: **anything interpolated must come from the file.**

**The ledger claimed work the indexes had not made.** A record asserts a
document is processed and the next build believes it — skipping the document,
so its units are never written. Committing that record before flushing the
indexes that hold the document makes the assertion a lie whenever the process
dies in between, and killing a build mid-run proved it: a ledger asserting 493
processed documents over indexes holding none, a resumed build that skipped all
493, and an arm reporting a plausible 0.952 failure rate instead of an error.

Builds now checkpoint — flush the indexes and unit store, *then* commit the
buffered ledger records. The unit of resumability becomes the checkpoint rather
than the document, which is the honest trade and is stated rather than implied.
A crash between the two costs a redundant reprocess: too much work, never too
little.

The general rule, for anything added later that persists build state: **flush
what holds the data, then record that you hold it.** The reverse ordering fails
in the one shape that is hardest to notice — the next run looks like it worked.

**A structured index answered text queries with arbitrary rows.** `search()`
ignored the query text and returned the first k units by id, scored 1.0 — the
same rows for every query, entering RRF with the weight of a genuine rank-1
hit. Not a weak signal but a *constant* one, displacing real top hits
identically across an entire query set. It surfaced in the no-router arm, where
every index is asked everything.

A structured index has no text ranking. With a filter it can still say which
rows match; with none there is neither constraint nor ranking, so it now returns
nothing. The contract note: **an index that cannot rank a query must return
empty rather than something.** An empty list costs a fuser nothing; a confident
wrong one costs it the top of the ranking.

**Per-stage accounting had 29% coverage, and that was the bug.** The manifest
summed 21s of stage time inside a 73s build and said nothing about the other
52s, because accounting only measured what was inside a stage. Printing the
residual (`BuildManifest.unaccounted_wall_ms`) found it immediately: the unit
store rewrote a 9 MB map once per document, and the ledger rewrote a 479 KB
snapshot once per document — the same quadratic-persistence defect the indexes
had, in two places the `Flushable` sweep did not cover because neither is an
`Index`.

Buffering the unit store took the build to 29.8s. Replacing the ledger's
rewrite with an append-only journal, compacted at `commit_build`, took it to
**24.8s — 2.95x faster — with accounting coverage at 86%**. The journal keeps
the durability guarantee exactly: a record is fsync'd the moment its document is
done, a load replays the journal over the snapshot, and a torn final line from a
killed process costs only the document it described.

The general lesson, and the reason the residual is now a manifest field: **an
accounting layer that measures only the work it knows about will report a
pipeline as fast while most of its time is somewhere else.** Invariant 6 asks
for per-stage cost and latency; it is worth nothing without a coverage number
next to it.

Two contract checks earned their place by catching bugs in the implementation
that proposed them: `check_parsed_document` and `check_units` found all of the
first two class of failures, on real documents, before any of it reached an
index.

### Round 2: an Italian company's archive

Pointing the frame at an Italian archive -- FatturaPA, PEC, signed files, scanned
contracts, folders with different readers, and an agent as the reader -- found
the next set. Each was reproduced before it was fixed, and each has a test that
fails without the fix.

**Identity leaked through the parse cache.** The cache is content-addressed,
and the cached value carried the first document's id and metadata; the same
contract attached to two emails collapsed into one document, and the second
tenant's copy was unreachable. Cached values are now stripped of identity and
rebound per document. The rule: **a content-addressed value must not contain
anything that is not content.**

**Filters were applied on some paths and not others.** The structured path ran
its query and ignored the caller's scope; with access control, that is a
tenant's question answered from every tenant's records. Scope, ACL and document
filters now apply on every path, and a test asks the structured path for what
it must not see.

**Italian was read as English.** Amounts ("1.250,00" became 1.25), dates
("30/06/2025" dropped), an ASCII tokenizer that split "città", a router whose
every rule was an English word. `indexer.normalize`, the Unicode tokenizer and
the Italian router are one fix each; the one that hid longest was the
analyser's per-unit language detection, which left one-line units -- most of an
invoice -- unstemmed while the questions about them were stemmed.

**Configuration that was accepted and never read.** `enrich.batch_size`,
`max_concurrency` and `on_error`; three segmenter limits; per-path rerankers;
`full.yaml` naming five implementations that did not exist while `check` said
OK; a manifest documented as written that never was. The rule: **a setting the
code does not read must fail validation, or it will be tuned for years.**

**Model output was taken on trust.** A refused or truncated answer arrived as
HTTP 200 and was cached as the answer. `indexer.llm` makes both errors, and the
field extractor keeps a value only when its quoted evidence is in the document
and states it -- a computed due date or an invented total goes to review, not
to a structured query that would compare against it.

**The router could not name what the archive held.** Its vocabulary came from
one enricher's params, so entity fields, FatturaPA facts and anything a model
extracted were fields no question could reach. Implementations now declare the
fields they write (`Registration.declares_fields`), and the router, the
config check and the agent's schema read that one list.

### Round 3: passages whose meaning is elsewhere

Round 3 found that invariant 3 had been read too narrowly. It also found seven
defects around it, several in the frame's own choices.

The trigger was one item of a building-services specification: "03.02.002
Idem c.s., ma per vuotatoi", the washbasin frame of the item above, for slop
sinks. An agent could not find it. The item was only an example of a problem
found everywhere, and the fix was made general:

* a reply that agrees to an option listed in the message it quotes;
* a clause whose party is named in the definitions and whose penalty is set
  eight articles away;
* a paragraph that opens with "Esso";
* a test report whose site is named only in the email it came with.

**Context can situate a chunk and still leave it unreadable.** The invariant
was implemented as situating context: what the document is, and where the
chunk sits in it. A statement that says "idem", "the latter" or "va bene,
procediamo" is not missing its place in the document. It is missing its
meaning, which is in other text, and no embedder can supply words a text does
not contain. Resolving that reference is a separate enrichment
(`llm_resolver`), with its own output: which statements depend on which
texts, and how they read with those filled in. It is checked the way extracted
fields are checked: the sources must be quoted, and every figure, name and
acronym must be found in them.

**The agent was handed the first 600 characters of a chunk.** On 25 chunkings
of a synthetic specification, the chunk holding the item ranked first 21 times.
The agent saw the item zero times, because it sat past the clip. The neighbour
text that `shape.expand_neighbors` attaches to every hit was computed and then
dropped by the agent's payload. Retrieval metrics scored all of this as
success. The metric to add beside fail@k: is the answer in what the reader is
given?

**Size thresholds decided that short entries were not worth finding.**
`merge_below_tokens` folds a short run into its neighbour on the grounds that
it is "too small to retrieve on". A ten-token entry is exactly as findable as
its reading makes it. So `items` never merges entries, and the size rules apply
only where size forced a cut. A document without numbered entries gets the
section rules unchanged, so one configuration serves an archive of both.

**The email parser threw away what replies answer.** It removed quoted history
so that a thread would not match every query its first message matches. That
was right for the index and wrong for the reader: "va bene, procediamo con la
seconda" lost the only text that says what the second option is. History is
now a `BlockKind.QUOTED` block. It stays in the document, and no segmenter may
make a unit of it or run a unit across it. So enrichers read it, and nothing
matches it.

**What an attachment means can be in another document, and one document's key
and ledger record could not say so.** An enricher that reads a second document
must key on it, and a change to that document must reach the first. Neither was
possible, because both the cache key and the ledger saw one document. Now:

* `ContextScope.RELATED` declares the dependency;
* `EnrichContext.related` supplies the linked documents, derived from the
  scanner's `PARENT_KEY`;
* the frame adds those documents to the enricher's key, whatever its own
  `input_hash` says;
* `DocumentRecord.related_hash` restages an attachment whose message changed
  while its bytes did not;
* a document is never read beside one with other readers, because a reading
  would carry that document's text to them.

**A resolver reads position, so its key must hold position.** "Idem c.s." means
whatever stands above it. Keyed on its text and document, two identical lines
under two different items shared one cache key. The pipeline makes one call
per distinct key, so both got the first line's reading. The key now includes
the span. The texts a reading draws on are stored as document offsets, not
unit ids, because offsets stay true when the document is segmented differently.

**Duplicates were decided on text alone.** The same "Idem c.s." line in two
specifications refers to two different items, but shaping collapsed the pair
into one hit. `shape.distinguish_by` names the enrichers whose context is part
of what a passage says.

**The first design sent every passage twice.** Each call carried the document,
then the batch's passages again, so an archive of short documents (most
emails, letters and invoices) paid about twice its size in input. Now the
passages are marked where they stand and named by position, so one marked-up
document serves every batch and is cached once. The instructions are a system
prompt that is the same for every call, so they are cached too. A document's
cache is written only when a later batch will read it.

---

## Evidence review, 2026-09-18

Every invariant was re-checked against 2025–2026 primary sources; the full
survey is `docs/STATE_OF_THE_ART.md`. The summary, for the maintainer who reads
only this file:

- **The contracts held.** Structure-first segmentation with zero overlap, the
  section-prefix control arm, rank-based fusion, a rules router as the
  reference, span-anchored gold and the retrieval failure rate are each the
  measured winner or the field's direction of travel in 2026. No contract needed
  changing.
- **Two invariants were over-stated in the prose.** Invariant 1's percentages
  come from an agent-memory study and are now scoped; invariant 5's "never" is
  now "never alone". The code already behaved the corrected way.
- **The reranker guidance was re-termed.** "Cross-encoder, not LLM" became
  "small pointwise on LOOKUP, listwise or reasoning only on ITERATIVE", because
  the best small rerankers are now LLM-based pointwise models.
- **Six seams were added to the table above** (tool-exposed retrieval, late
  chunking, visual, tree, learned sparse, graph), each with the evidence that
  earns or withholds a default.
- **Two ablation arms were added** to `configs/reference.yaml`: RRF `k=20` and
  `overlap_tokens: 64`, so the two most-cited 2026 fusion and chunking findings
  can be checked on any corpus.
- **The first real-corpus ablation was run**; see `docs/ABLATION-pypi-docs.md`.

---

## What would falsify this design

Written down so it is checkable rather than a matter of taste.

1. **Adding a fifth index kind requires editing `indexer/core`.** Then the
   registry indirection failed at its one job.
2. **Two arms of an ablation differ in a way the override list does not
   state.** Then the delta table is not attributable and invariant 6 is
   unsupported.
3. **A golden set stops working after a segmenter change.** Then span anchoring
   failed and no two arms are comparable.
4. **The sanity checks fail on a real corpus and the cause is in the frame
   rather than the corpus.** Most likely: an index indexing `unit.text` instead
   of `indexing_text()`. `check_index_surface()` exists to catch exactly this.
5. **A new project needs a fork.** Then the config schema is too closed, and the
   place to look is phase-1 validation having grown an enumeration of
   implementations.

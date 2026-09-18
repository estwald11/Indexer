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
config rather than discovered during a rebuild.

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
| Agent framework | `Retriever` on the `ITERATIVE` path, under `step_budget`. `Query.context` already carries prior turns; the frame does not manage dialogue but will not drop it. |
| Tool-exposed retrieval | The 2025–26 agentic-search results (Anthropic's context-engineering guidance; *Keyword search is all you need*, AAAI 2026) want each index callable as a *tool* by a budgeted loop, not only as one fused pass. Each `Index.search` is already that call; what is missing is a thin tool adapter over `QueryEngine` that keeps `step_budget` and the decision log. |
| Vendor coupling | Every vendor sits behind `Registration` and an extra. `indexer.core` imports nothing third-party. |
| Distributed indexing | `Ledger` and `CacheStore` are protocols; a distributed build needs a shared ledger with per-document locking and a shared cache. `PlannedChange` is already a partitionable work list. |
| Quantization | Inside a dense `Index` implementation. `IndexStatsView.detail` carries the knobs; the Matryoshka note in `configs/full.yaml` is there so the two-stage option is not foreclosed. HAKARI-Bench (2026): int8 with float rescoring is lossless in practice; binary with rescoring costs ≈1 point for 32× less storage. |
| Late chunking / contextual embeddings | Inside the dense `Index`: the embedder sees the whole document and returns one vector per unit. `Index.upsert` already receives units batched by document. |
| Visual (page-image) index | The `visual` kind in `configs/full.yaml`. ViDoRe v3 (2026) says: route per page, fuse with the text legs, and rerank with a *text* reranker; store pooled, truncated or binarised multi-vectors. |
| Tree / table-of-contents index | A `tree` kind: TOC tree with node summaries (PageIndex-style, ~$0.001/page), plus a section-as-file layout of `ParsedDocument` so grep-style agents can navigate at zero model cost. Routed to for long single-document analytical questions. |
| Learned sparse leg | A third first-stage index (`SPLADE-v3`, OpenSearch neural-sparse v3-gte) behind the same `Index` protocol. Must earn its place: a weak leg drags fusion down. |
| Graph enrichment | An optional `Enricher` writing entities and relations into `Enrichment.extra`, gated on a multi-hop-heavy query mix. Fair 2026 benchmarks (GraphRAG-Bench, WildGraphBench) show it losing to hybrid + rerank on fact retrieval. |
| Deletion semantics | `Index.delete` says "must actually remove". *Ghost Vectors* (Jun 2026) shows soft-deleted vectors stay recoverable in HNSW files, and most stores compact lazily. The manifest should record the store's compaction policy; a compaction or key-rotation step belongs after any bulk re-index. |
| Embedding-model migration | Drift-Adapter (EMNLP 2025) and shared-space model families (Voyage 4) make "swap the model without re-embedding everything" possible. The gate is the eval harness: 200–500 labelled queries before cutover. |
| Multi-tenancy | `Predicate` filters push down to every index, and `SourceSpec.namespace` scopes document ids. What is missing is per-tenant index isolation, which is an `Index` implementation concern. |

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

Two contract checks earned their place by catching bugs in the implementation
that proposed them: `check_parsed_document` and `check_units` found all of the
first two class of failures, on real documents, before any of it reached an
index.

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

# Ablation report: PyPI documentation corpus

What this run establishes, what it does not, and what it changed about the
contracts. Read the caveats before the table — two of the four published
expectations did not reproduce, and the reasons are more useful than the
numbers.

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
text, so every query is a bag of words that literally occurs in the passage it
is looking for. BM25 is therefore scored on precisely what it does, and no query
in the set *requires* matching meaning rather than words. This is not a flaw in
the generator — it was built to test contextualisation, and for that the
construction is right — but it is a ceiling on the dense arms, and it binds a
neural bi-encoder exactly as hard as it binds the two embedders measured here.

**The golden set is machine-generated and unverified.** 333 queries from
`HeuristicBootstrapper`, filtered against a lexical baseline: 1,162 candidates
dropped as too easy (the baseline already ranked the gold first, so they cannot
distinguish two systems) and 48 as unfindable. 300 factual, 16 structured, 13
temporal, 4 numeric. No human has checked them.

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
ids, which is what lets the same 333 queries score the structural segmenter and
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
seeded, fusion breaks ties by unit id, and re-running reproduced every arm's
numbers exactly.


## The embedder, measured

`2-dense-only` and `3-hybrid-rrf` were read, when this report was first written,
as evidence that dense retrieval does not work on this corpus. That reading
conflated two claims: *dense retrieval does not help here*, and *this particular
projection does not work*. Splitting the embedder out of the store
(`impls/embed.py`) made them separable — `2b-svd-only` and `3c-hybrid-svd`
differ from their counterparts in the embedder key and nothing else.

| arm | P@5 | R@20 | nDCG@10 | fail@20 | p50 |
|---|---|---|---|---|---|
| `1-lexical-only` | 0.145 | 0.917 | 0.573 | **0.075** | 8ms |
| `2-dense-only` (hash) | 0.054 | 0.473 | 0.209 | 0.474 | 5ms |
| `2b-svd-only` (LSA) | 0.100 | 0.797 | 0.390 | **0.183** | 21ms |
| `3-hybrid-rrf` (hash) | 0.105 | 0.847 | 0.411 | 0.138 | 17ms |
| `3c-hybrid-svd` (LSA) | 0.136 | 0.887 | 0.529 | **0.102** | 35ms |

**Most of the dense arm's failure was the embedder: 47.4% to 18.3%, a 61%
reduction, with the store, the fusion, the golden set and the matcher all
unchanged.** Fused, 13.8% to 10.2%, and nDCG@10 from 0.411 to 0.529 — the
largest single-change improvement any arm in this report has produced. Every
pre-existing arm reproduced its numbers exactly across the refactor, which is
the only reason those rows are comparable at all.

Three things this does *not* establish.

*Invariant 4 still does not reproduce.* The hybrid remains worse than its
lexical half, 10.2% against 7.5%. The gap narrowed from +84% to +36%, which is
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

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

**The dense index is not a semantic model.** No credentials and no model
downloads were available in this environment (`huggingface.co` is blocked by the
network policy; there is no `ANTHROPIC_API_KEY`), so the dense arm is
`hash_embedding` — a hashing trick over word and character n-grams. It is a real
vector index with real write, delete and filter semantics, and it is *not* an
embedding model: it cannot match "how do I stop a request hanging" to "timeout"
unless the words overlap. Treat every absolute number below as
uninterpretable and every delta as the finding.

**The contextualiser is extractive, not an LLM.** Same reason. This turns out to
matter more than expected — see the first finding.

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

Ten arms. Each is an override list against one config, so two arms provably
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


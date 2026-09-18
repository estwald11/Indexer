# indexer

A document indexing and retrieval frame. Eight stages, each an interface with a
stated contract; every concrete choice behind a registry and named from a config
file.

It is built on the assumption that **every implementation in it will be replaced
at least once**. What is meant to survive is the shape: the stage boundaries, the
types that cross them, the content-addressed caching underneath them, and the
evaluation harness that decides whether a replacement was an improvement.

```
parse -> segment -> enrich -> index      (ingestion, paid once)
route -> retrieve -> fuse -> rerank      (query, paid forever)
```

## Status

**Round 2 complete; round 3 (2026-09-18) is an evidence review.** The frame,
two reference implementations per stage, the orchestrators and the eval harness
exist and pass their tests offline. Round 3 re-checked every invariant against
2025–2026 primary sources and ran the first ablation on a real corpus. What
exists:

| | |
|---|---|
| `src/indexer/core/` | The eight stage protocols and the types that cross them. Dependency-free. |
| `src/indexer/config/` | The config schema, overlays, interpolation, and two-phase validation. |
| `src/indexer/impls/` | Reference implementations, at least two per stage, all runnable offline. |
| `src/indexer/pipeline/` | The ingestion and query orchestrators that enforce the contracts. |
| `src/indexer/eval/` | Golden-set format, metrics, bootstrapper, ablation runner, contract checks. |
| `configs/reference.yaml` | The deliberately simple path, plus the ablation ladder. |
| `configs/full.yaml` | The same frame with production-shaped choices — as an overlay, to show that it is only config. |
| `configs/pypi-docs.yaml` | The offline ablation corpus: documentation from 50 PyPI packages. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Which invariant drove which contract, and what to revisit when the evidence changes. |
| [`docs/STATE_OF_THE_ART.md`](docs/STATE_OF_THE_ART.md) | The 2026 survey: what the published evidence says about every stage, and where it attaches to the frame. |
| [`docs/ABLATION-pypi-docs.md`](docs/ABLATION-pypi-docs.md) | The first ablation run on a real corpus, with the sanity-check verdicts. |
| [`docs/REVIEW.md`](docs/REVIEW.md) | The decisions worth arguing about, the rulings, and the round-3 changes. |

## The six invariants

These are evidence-backed, and they shape the frame rather than sitting in a
doc. `ARCHITECTURE.md` traces each one to the contract it forced;
`docs/STATE_OF_THE_ART.md` §0 audits each against the 2025–2026 literature.

1. **Retrieval is the bottleneck, not generation.** Oracle-versus-retrieved gaps
   of 7–30 points recur across 2025–26 document benchmarks. The often-quoted
   split — retrieval failures 11–46% of questions, utilisation failures 4–8%,
   Precision@5 correlated with accuracy at r=0.98 — is one 2026 agent-memory
   study's result on nine configurations, and is directional rather than a law.
2. **Ingestion cost is paid once, query cost forever.** Push work upstream.
3. **Chunks must carry their context.** Anthropic's figures: a 50–100 token
   summary prepended before both embedding and lexical indexing, 5.7% → 2.9%
   top-20 retrieval failure; reranking → 1.9%. Independent replications find
   smaller gains at real compute cost, and one 2026 study finds context can hurt
   within-document questions. The size of the effect is measured per corpus,
   not assumed.
4. **Hybrid beats either half.** Still true with 2026 embedders (+3–5 nDCG
   points over dense alone on every model tested), with two caveats: a weak leg
   drags fusion down, and reranker depth has to be bounded.
5. **Structured, numeric and temporal questions go to a structured executor,
   never to vector search alone.** SQL over extracted fields beats
   flattened-table retrieval on aggregation; text retrieval is still what
   locates the table, so the structured path keeps a lookup fallback.
6. **Nothing is optimized without a before/after number.** And no number
   without a confidence interval.

## Reading order

Start with `ARCHITECTURE.md`. Then `src/indexer/core/stages.py`, which is the
whole frame: eight protocols, each stating its input and output, what it may
assume, what it must preserve, and what its minimal implementation looks like.
Then `docs/STATE_OF_THE_ART.md` for what the evidence says each implementation
should be in 2026, and `docs/ABLATION-pypi-docs.md` for what the harness
measured on a real corpus.

## Development

```bash
uv venv && uv pip install -e ".[dev,fast]"    # or: python -m venv .venv && pip install -e ".[dev,fast]"
pytest                                    # contract and pipeline tests
ruff check src tests scripts && mypy      # lint and types
python -m indexer.config.check configs/reference.yaml   # validate a config
python scripts/fetch_corpus.py            # the offline ablation corpus (PyPI docs)
python scripts/run_ablation.py configs/pypi-docs.yaml   # build, bootstrap gold, run the ladder
```

Python 3.11 or newer. The reference path needs no model download and no API key.

## Out of scope

No UI, no agent framework, no vendor coupling, no distributed indexing, no
quantization. Each has a documented seam in `ARCHITECTURE.md` under
[Seams](ARCHITECTURE.md#seams-left-open).

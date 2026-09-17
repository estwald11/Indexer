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

Contracts, reference implementations and the evaluation harness are in. Every
stage has at least two implementations selected by config; nothing in the frame
names one.

| | |
|---|---|
| `src/indexer/core/` | The eight stage protocols and the types that cross them. Dependency-free. |
| `src/indexer/config/` | Config schema, overlays, interpolation, two-phase validation. |
| `src/indexer/pipeline/` | Ingestion and query orchestrators. Where the contracts are enforced. |
| `src/indexer/impls/` | Reference implementations, two or more per stage. |
| `src/indexer/eval/` | Golden-set format, metrics, bootstrapper, ablation runner, contract checks. |
| `configs/reference.yaml` | The deliberately simple path. Runs offline, no credentials. |
| `configs/full.yaml` | Production-shaped choices, as an overlay — to show it is only config. |
| `configs/pypi-docs.yaml` | The ablation corpus and its eight-arm ladder. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Which invariant drove which contract, and what implementing it changed. |
| [`docs/ABLATION.md`](docs/ABLATION.md) | The ablation report on a real corpus, and what it does and does not establish. |
| [`docs/REVIEW.md`](docs/REVIEW.md) | Decisions taken, and the environment constraints on the report. |

```
                     ingestion (paid once)
   corpus ──▶ parse ──▶ segment ──▶ enrich ──▶ index
                 │          │          │          │
             ParsedDoc    Unit    EnrichedUnit   ...indexes
                                                    │
   query ──▶ route ──▶ retrieve ──▶ fuse ──▶ rerank ┘
                          query (paid forever)
```

## Quick start

```bash
uv venv && uv pip install -e ".[dev,fast]"
python scripts/fetch_corpus.py                        # 686 docs from 50 PyPI packages
python -m indexer.config.check configs/pypi-docs.yaml # validate without importing impls
python scripts/run_ablation.py configs/pypi-docs.yaml # build, bootstrap, ablate, report
```

Everything runs offline: no API keys, no model downloads. See
[`docs/REVIEW.md`](docs/REVIEW.md) for what that costs — in short, the reference
dense index is a hashing trick rather than a semantic model, so absolute numbers
from it are not comparable to published figures. The deltas are the point.

## The six invariants

These are evidence-backed, and they shape the frame rather than sitting in a
doc. `ARCHITECTURE.md` traces each one to the contract it forced.

1. **Retrieval is the bottleneck, not generation.** Retrieval failures drive
   11–46% of end-to-end errors; utilisation failures stay at 4–8%. Precision@5
   predicts answer accuracy at r=0.98.
2. **Ingestion cost is paid once, query cost forever.** Push work upstream.
3. **Chunks must carry their context.** A 50–100 token summary prepended before
   both embedding and lexical indexing: 5.7% → 2.9% top-20 retrieval failure.
   Reranking: → 1.9%.
4. **Hybrid beats either half.** Dense plus sparse, fused, then reranked.
5. **Structured, numeric and temporal questions must never reach vector search.**
6. **Nothing is optimized without a before/after number.**

## Reading order

Start with `ARCHITECTURE.md`. Then `src/indexer/core/stages.py`, which is the
whole frame: eight protocols, each stating its input and output, what it may
assume, what it must preserve, and what its minimal implementation looks like.

## Development

```bash
pytest                       # 100 tests: contracts, config, pipeline, SQL conformance, layering
ruff check src tests && mypy # lint and strict types
```

Three layering rules are enforced by tests rather than convention:
`indexer.core` imports nothing third-party and names no implementation, and
`indexer.eval` depends on no implementation — so the harness can evaluate a
system this library did not build.

## Out of scope

No UI, no agent framework, no vendor coupling, no distributed indexing, no
quantization. Each has a documented seam in `ARCHITECTURE.md` under
[Seams](ARCHITECTURE.md#seams-left-open).

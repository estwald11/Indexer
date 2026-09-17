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

**Round 1: contracts and config schema, for review.** No implementations yet —
that is deliberate, and the review gate is the point. What exists:

| | |
|---|---|
| `src/indexer/core/` | The eight stage protocols and the types that cross them. Dependency-free. |
| `src/indexer/config/` | The config schema, overlays, interpolation, and two-phase validation. |
| `src/indexer/eval/` | Golden-set format, metrics, the ablation delta table, contract checks. |
| `configs/reference.yaml` | The deliberately simple path, plus the ablation ladder. |
| `configs/full.yaml` | The same frame with production-shaped choices — as an overlay, to show that it is only config. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Which invariant drove which contract, and what to revisit when the evidence changes. |
| [`docs/REVIEW.md`](docs/REVIEW.md) | The decisions worth arguing about before implementations get written. |

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
uv venv && uv pip install -e ".[dev]"
pytest                                    # contract tests
ruff check src tests && mypy              # lint and types
python -m indexer.config.check configs/reference.yaml   # validate a config
```

## Out of scope

No UI, no agent framework, no vendor coupling, no distributed indexing, no
quantization. Each has a documented seam in `ARCHITECTURE.md` under
[Seams](ARCHITECTURE.md#seams-left-open).

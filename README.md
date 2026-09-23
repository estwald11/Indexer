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
| `src/indexer/impls/embed.py` | The embedder seam: hashing, LSA, and a neural bi-encoder behind one store. |
| `src/indexer/eval/` | Golden-set format, metrics, bootstrappers, ablation runner, contract checks, significance, tuning. |
| `src/indexer/llm.py` | One way to call Claude for every model-backed stage: priced, refusal- and truncation-aware, structured outputs, batchable. |
| `src/indexer/agent.py` | The archive as an agent's tool set; `mcp_server.py` serves it over MCP, `cli.py` is the `indexer` command. |
| `configs/reference.yaml` | The deliberately simple path. Runs offline, no credentials. |
| `configs/full.yaml` | Production-shaped choices, as an overlay — to show it is only config. |
| `configs/it-enterprise.yaml` | An Italian company's archive: FatturaPA, PEC, Office, ACLs, Italian analysis, an agent's tools. |
| `configs/pypi-docs.yaml` | The ablation corpus and its eight-arm ladder. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Which invariant drove which contract, and what implementing it changed. |
| [`docs/ABLATION.md`](docs/ABLATION.md) | The ablation report on a real corpus, and what it does and does not establish. |
| [`docs/STATE_OF_THE_ART.md`](docs/STATE_OF_THE_ART.md) | The 2026 survey: what the published evidence says about every stage, and where it attaches to the frame. |
| [`docs/ABLATION-pypi-docs.md`](docs/ABLATION-pypi-docs.md) | The first, eight-arm run on the same corpus, kept for the record; superseded by `ABLATION.md`. |
| [`docs/REVIEW.md`](docs/REVIEW.md) | Decisions taken, the environment constraints on the report, and each round's changes. |

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

Everything above runs offline: no API keys, no model downloads. See
[`docs/REVIEW.md`](docs/REVIEW.md) for what that costs — in short, the reference
dense index is a hashing trick rather than a semantic model, so absolute numbers
from it are not comparable to published figures. The deltas are the point.

The dense index has three embedders behind one store, which is the seam a real
model drops into:

| `impl` | What it is | Needs |
|---|---|---|
| `hash_embedding` | Hashing trick. Fixed projection, offline, the CI default. | — |
| `svd_embedding` | LSA: TF-IDF then truncated SVD. *Fitted* on the corpus, still offline. | `[dense-svd]` |
| `sentence_transformer` | A real bi-encoder (BGE/E5/MiniLM). The production choice. | `[dense-sentence-transformer]` |
| `multilingual_embedding` | bge-m3 or multilingual-e5 by preset, with each model's own prefixes. For Italian. | `[dense-sentence-transformer]` |
| `voyage_embedding` | Voyage AI's API. Nothing to host; every chunk goes to a third party. | `VOYAGE_API_KEY` |

Swapping the first for the second cut dense-only retrieval failures by 61% with
nothing else changed — see [the embedder section](docs/ABLATION.md#the-embedder-measured)
for what that does and does not establish.

### What a golden set can measure

Every generated item records `lexical_overlap`: how much of the query is lifted
verbatim from the passage it is looking for. Report it beside any retrieval
number, because near 1.0 a set scores a word matcher on exactly what it does and
contains few items that *require* matching meaning — which caps every dense arm
evaluated against it, a neural bi-encoder included. The shipped PyPI set sits at
0.75, and its detail terms at 0.91.

| `impl` | What it generates | Needs |
|---|---|---|
| `heuristic` | Subject + detail terms lifted from the unit. Offline, the CI default. | — |
| `llm_bootstrap` | The same items rewritten as questions that avoid the passage's wording. | `anthropic` |

Surface transforms were tried first as a free substitute and measured: they make
every arm worse without changing their order. The table is in
[`docs/ABLATION.md`](docs/ABLATION.md#paraphrasing-the-set-what-was-tried-and-what-it-cost).

With `language: it` the structured questions are written in Italian, worded so
the rules router reads each as the comparison it states, and `llm_bootstrap`
rewrites its items in Italian.

### Whether a difference is a difference

The delta table pairs every arm with the baseline query by query: a bootstrap
95% interval for graded metrics (nDCG, recall, MRR) and McNemar's test for
pass/fail ones (retrieval failure, correctness), marked `*` when significant.
`GoldenSet.split` divides a set into dev and test by a hash of each query id, so
adding queries never moves one across; `run_ablation.py --tune-fusion` fits RRF
weights — overall and per predicted query type — on dev and reports them on
test beside the default. `eval.agent_metrics` scores what an agent is actually
handed: whether the answer is among the tool's results, whether every citation
resolves, how large the payload is.

## An Italian company's archive

`configs/it-enterprise.yaml` is the starting point, and `indexer check` resolves
every name in it:

```bash
pip install -e ".[office,analysis,llm,mcp,dense-sentence-transformer]"
export ARCHIVE_ROOT=/mnt/archivio ARCHIVE_STATE=/var/lib/archivio
indexer check   configs/it-enterprise.yaml
indexer prefill configs/it-enterprise.yaml   # the first build's model calls by batch, at half price
indexer build   configs/it-enterprise.yaml
indexer query   configs/it-enterprise.yaml "fatture di Rossi sopra i 1.000 euro nel 2024" --principal group:amministrazione
indexer mcp     configs/it-enterprise.yaml --principal group:amministrazione
```

What it does with the archive:

* **Opens what is stored.** Signed files (`.p7m`, CAdES), email and PEC with
  their attachments and `daticert.xml`, zip archives; FatturaPA read as exact,
  typed facts; Word, Excel, HTML and text-layer PDF with their structure. Folder
  rules and sidecar files set each document's ACL.
* **Reads Italian.** BM25 with accent folding, Italian stopwords, elisions and
  stemming, the document's language rather than a one-line unit's guess; numbers
  and dates as Italians write them ("1.250,00", "30/06/2025", "giugno 2025").
* **Knows the things documents are about.** VAT numbers, fiscal codes and IBANs
  kept only when their check digit holds; optional joins to the company's
  customer and supplier registers.
* **Uses a model where one earns it, and checks it.** A classifier (one call per
  document), a per-type field extractor whose every value must be quoted from
  the document or it goes to `indexer review`, LLM contexts written in the
  document's language, and a router that lets the rules answer what they can.
  Each is optional, priced in the manifest, and answered through the Message
  Batches API by `indexer prefill`.
* **Serves an agent.** `search`, `query_records`, `describe_schema`,
  `get_document`, `outline`, `expand` and `find_entity` — with citations, the
  build they reflect, the caller's access rights on every one, and archive text
  marked as data rather than instructions. Over MCP, or in-process through
  `indexer.agent.AgentTools`.

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
pytest                       # contracts, config, pipeline, durability, embedders,
                             # golden sets, SQL conformance, layering, LLM stages
                             # (against a fake client: no network, no key)
ruff check src tests scripts && mypy   # lint and strict types
```

Three layering rules are enforced by tests rather than convention:
`indexer.core` imports nothing third-party and names no implementation, and
`indexer.eval` depends on no implementation — so the harness can evaluate a
system this library did not build.

## Out of scope

No UI, no agent loop (the tools an agent calls are here; deciding when to call
them is the agent's), no vendor coupling, no distributed indexing, no
quantization. Each has a documented seam in `ARCHITECTURE.md` under
[Seams](ARCHITECTURE.md#seams-left-open).

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
| `configs/it-enterprise.yaml` | An Italian company's archive: FatturaPA, PEC, Office, ACLs, Italian analysis, passages read out where their meaning is elsewhere, an agent's tools. |
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
  typed facts; Word, Excel, HTML and text-layer PDF with their structure, one
  unit per numbered entry where a document is made of them. Folder rules and
  sidecar files set each document's ACL.
* **Reads Italian.** BM25 with accent folding, Italian stopwords, elisions and
  stemming, the document's language rather than a one-line unit's guess; numbers
  and dates as Italians write them ("1.250,00", "30/06/2025", "giugno 2025").
* **Knows the things documents are about.** VAT numbers, fiscal codes and IBANs
  kept only when their check digit holds; optional joins to the company's
  customer and supplier registers.
* **Uses a model where one earns it, and checks it.** A classifier (one call per
  document), a per-type field extractor whose every value must be quoted from
  the document or it goes to `indexer review`, LLM contexts written in the
  document's language, a resolver for passages whose meaning is written
  elsewhere (below), and a router that lets the rules answer what they can.
  Each is optional, priced in the manifest, and answered through the Message
  Batches API by `indexer prefill`.
* **Serves an agent.** `search`, `query_records`, `describe_schema`,
  `get_document`, `outline`, `expand` and `find_entity` — with citations, the
  build they reflect, the caller's access rights on every one, and archive text
  marked as data rather than instructions. Over MCP, or in-process through
  `indexer.agent.AgentTools`.

## Passages whose meaning is written elsewhere

A passage is indexed by its words, and many passages mean more than they say.
They take their subject, their object or their whole content from other text:

| Document | The passage says | What it means is in |
|---|---|---|
| Specification | "03.02.002 Idem c.s., ma per vuotatoi" | the item above: the washbasin frame, now for slop sinks |
| Email reply | "va bene, procediamo con la seconda soluzione" | the message it quotes, which lists the two options |
| Contract | "L'Appaltatore risponde dei ritardi nei termini dell'art. 12" | art. 1, which names the contractor, and art. 12, which sets the penalty |
| Report | "Esso dovrà essere sottoposto a manutenzione semestrale" | the section before, which describes the 250 kW chiller |
| Attachment | "La prova di tenuta è stata eseguita a 6 bar ... con esito positivo" | the email it came with, which names the site and the system |

No index finds any of these by what it means, lexical or dense, however good
the embedder, because the words are not in the passage. Invariant 3's situating
context does not close the gap either: it says where a chunk sits, not what a
statement refers to.

What the frame does about it:

* **A model reads such statements out** (`enrich: llm_resolver`). For each
  passage it lists every statement that takes its meaning from outside the
  passage, with the texts it takes it from, and writes the statement out as it
  reads with those filled in. Every passage gets a verdict, so none is skipped
  by omission.
* **Each reading is checked before it is indexed.** The statement must be
  quoted from its passage and each source from the document. Every figure,
  name and acronym in the reading must occur in those sources, and nearly all
  its other words too. A reading that fails goes to `indexer review`. One that
  passes joins every index's retrieval surface. The passage's own text is never
  changed.
* **It reads what the statement reads.** That is the document, with its
  passages marked where they stand, and the history a reply quotes. The email
  parser keeps that history as quoted context: it is in no unit, so a thread
  does not match every query its first message matches, but the resolver sees
  it. With `relations: [container]` it also reads the message an attachment
  came with (`ContextScope.RELATED`). A changed message restages its
  attachments. A document is never read beside one with other readers.
* **Entries are units** (`segment: items`). A document made of numbered entries
  (a specification, a price list, a bill of quantities, a numbered procedure)
  gets one unit per entry, however short. Every other document is cut at its
  sections.
* **The agent is shown the reading.** `search` and `expand` return it as
  `resolved`, marked as a model's reading, beside the texts it draws on: a
  passage, a quoted message, or another document. A reading is withheld when
  the caller may not see one of those texts. Two passages with the same words
  but different readings are no longer collapsed as duplicates
  (`shape.distinguish_by`), and neighbouring text reaches the agent.

Measured on the five documents above (`tests/test_resolver.py`), this is the
rank of each passage for a query about what it means, under BM25:

| Passage | Query | Before | After |
|---|---|---|---|
| Idem c.s., ma per vuotatoi | telaio autoportante per vuotatoio | 2 | 1 |
| L'Appaltatore risponde ... art. 12 | penale per i ritardi di Rossi Impianti | 3 | 1 |
| Esso dovrà essere sottoposto ... | manutenzione del gruppo frigorifero da 250 kW | 2 | 1 |
| va bene, procediamo con la seconda | pompa di calore Mitsubishi | not found | 1 |
| La prova di tenuta ... | prova di tenuta impianto idrico-sanitario Via Roma 10 | 2 | 1 |

This shows the mechanism works when the readings are correct: those readings
were written by hand. It does not show how good a model's readings are. That
needs an API key, a real archive, and a golden set of such statements written
by hand, since a generated set draws each query from the passage's own words
and cannot ask for what a passage does not say. `it-enterprise.yaml` carries
four arms to measure it:

* `senza-risolutore`;
* `risolutore-senza-allegati`;
* `risolutore-sonnet`;
* `segmenti-per-sezione`.

What it costs: the model's instructions are cached across calls, and each
document is sent once. A long document is split into batches, and its later
batches read it from the cache at a tenth of the input price. The input is
therefore close to the archive's own size in tokens, paid once per version of
each document: at Opus 5's list price, $5 per million tokens, or roughly 1,500
dense pages. `indexer prefill` halves it. Output depends on how many
statements need reading and how much the model thinks, and the manifest
reports both. Price a sample before a whole archive.

Adapting it to a domain is configuration:

* its ways of referring go in the resolver's `instructions`, for example "c.s.
  vuol dire come sopra" or "wie vor";
* its entry codes go in `segment.params.item_pattern` when the default does not
  know them;
* its documents that need no reading go in `skip_when`.

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

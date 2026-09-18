# State of the art: indexing large documents for LLM retrieval

**As of 2026-09-18.** A survey of the published evidence behind each stage of
the frame, written for the maintainer who has to decide what to swap, what to
add and what to leave alone. Every number is tied to a primary source. Where a
figure could only be found in a vendor's own benchmark it is marked
**[vendor]**; where it could not be read from the primary source at all it is
marked **[unverified]** and should not be built on.

The document is organised by stage so that a finding lands next to the seam it
would attach to:

```
parse -> segment -> enrich -> index      (ingestion, paid once)
route -> retrieve -> fuse -> rerank      (query, paid forever)
```

Each finding ends with an *implication* tagged as one of three kinds of change
the frame allows without a fork: **swap** (a different implementation behind
an existing registry name), **capability** (a new protocol or index kind), or
**knob** (a config key).

---

## 0. Claim audit: the six invariants against the evidence

The README and `ARCHITECTURE.md` rest on six invariants that were written from
a 2024 brief. This is what a 2026 re-check found.

| # | Invariant as originally stated | Primary source | Verdict | Corrected reading |
|---|---|---|---|---|
| 1 | Retrieval failures drive 11–46% of end-to-end errors; utilisation failures stay at 4–8%; Precision@5 predicts accuracy at r=0.98 | Yuan, Su, Yao, *Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory*, [arXiv 2603.02473](https://arxiv.org/abs/2603.02473), Mar 2026 | **Numbers correct, scope overstated.** The study is about *agent memory* on one conversational benchmark (LoCoMo, 1,540 questions, nine configurations, one reader model). "Retrieval failure" there includes write-side failures. r=0.98 is a correlation across nine points. | Retrieval is the dominant failure mode: oracle-vs-retrieved gaps of 7–30 points recur across 2025–26 document benchmarks (T²-RAGBench, Akarsu et al.). The specific percentages are one study's, not a law. Precision@5 stays the headline metric because it is cheap and directionally right, not because of r=0.98. |
| 2 | Ingestion cost is paid once, query cost forever | — (engineering principle) | **Holds.** Reinforced by 2026 work on embedding migration (Drift-Adapter) and by the fact that reranking, not indexing, is where marginal compute pays. | Unchanged. |
| 3 | A 50–100 token LLM summary prepended before both embedding and lexical indexing cuts top-20 failure 5.7% → 2.9%; reranking → 1.9% | Anthropic, [*Introducing Contextual Retrieval*](https://www.anthropic.com/engineering/contextual-retrieval), Sep 2024 | **Figures accurate, but vendor-internal.** Independent replications find smaller gains (≈ +6% nDCG@10 on NFCorpus; Merola & Singh 2025) at real compute cost, and Zhou et al. (Feb 2026) find contextualisation *degrades in-document retrieval* while improving in-corpus retrieval. | Context helps; the magnitude is corpus-dependent and the direction is not guaranteed for within-document questions. The sanity check in `configs/*.yaml` should be read as "does it move the right way", not "does it hit −⅓". |
| 4 | Hybrid beats either half | Vespa, Jan 2026; Akarsu et al., Apr 2026; TREC 2025 RAG overview; *contra* Wang et al. 2025, Jacob et al. 2025 | **Holds as a default, with two caveats.** Fusion is only as good as its weakest leg, and reranker quality *declines* past a depth threshold. | Run both legs and fuse; validate each leg per corpus; bound reranker depth. |
| 5 | Structured, numeric and temporal questions must never reach vector search | TableRAG (EMNLP 2025); T²-RAGBench (EACL 2026); Akarsu et al. 2026; RAGRouter-Bench 2026 | **Direction supported, "never" overstated.** SQL execution over extracted tables beats flattened-table RAG on aggregation and nested questions; text or hybrid retrieval is still what *locates* the table. | Route aggregation, comparison and date-range questions to the structured executor; keep lookup as the fallback when no predicate is extractable (which is what `RulesRouter` already does). |
| 6 | Nothing is optimised without a before/after number | TREC 2025 RAG; UMBRELA; Clarke & Dietz 2026; BRIGHT; FreshStack | **Holds, and the field moved towards it.** Nugget- and span-anchored gold is now the direction of TREC, FreshStack and ViDoRe v3. LLM judges rank *systems* reliably (Kendall τ 0.87–0.94) but agree with humans on *items* poorly (κ ≈ 0.3). | Unchanged. Add paired-bootstrap confidence intervals and keep a human-verified subset. |

One more claim in the code comments deserves its own line. *"Avoid LLM-as-reranker
on the simple path; use a cross-encoder"* is still sound advice, but the
terminology is dated: Qwen3-Reranker-0.6B is an LLM-based *pointwise* reranker
that sits exactly where a cross-encoder used to. The distinction that matters
in 2026 is **small pointwise** (cheap, use on LOOKUP) versus **listwise or
reasoning** (expensive, use on ITERATIVE).

---

## 1. Parse

### 1.1 What the leaderboards say

**OmniDocBench** (OpenDataLab, CVPR 2025; v1.6 since 2026-04-30) remains the
reference for end-to-end page parsing. Top of the table as of 2026-09-11
([GitHub](https://github.com/opendatalab/OmniDocBench)):

| Model | Overall | Params | Table TEDS | Reading-order edit ↓ |
|---|---|---|---|---|
| TeleOCR | 96.91 | 1.2B | 96.82 | 0.118 |
| OvisOCR2 | 96.47 | 0.8B | 94.58 | 0.112 |
| PaddleOCR-VL-1.6 | 96.34 | 0.9B | 94.76 | 0.128 |
| MinerU2.5-Pro | 95.75 | 1.2B | 93.42 | 0.120 |
| GLM-OCR | 95.22 | 0.9B | 92.83 | 0.133 |

Every entry in the 2026 top ten is an end-to-end vision-language model under
2B parameters. MinerU2.5-Pro's own paper attributes its gain purely to data
engineering (10M → 65.5M training samples) and notes it "surpasses models with
over 200× more parameters" ([arXiv 2604.04771](https://arxiv.org/abs/2604.04771)).

Three 2026 benchmarks qualify that picture:

- **PureDocBench** ([arXiv 2605.07492](https://arxiv.org/abs/2605.07492)) audited
  OmniDocBench and found 2,580 annotation errors (12.08% of blocks). On its own
  clean/degraded/real-degraded set the best of 40 models scores ~74/100, and no
  model exceeds 67% on formulas.
- **Real5-OmniDocBench** ([arXiv 2603.04205](https://arxiv.org/abs/2603.04205))
  re-photographs every page under scan, warp, screen, illumination and skew;
  "the reality gap is far from closed".
- **RealDocBench** ([arXiv 2606.07401](https://arxiv.org/abs/2606.07401)) scores 18
  parsers on 581 regulated documents by *field-level QA*, not edit distance, and
  finds single-number benchmarks hide the differences that matter.

The other harness in wide use is **olmOCR-Bench** (Ai2, unit-test style). README
top: Chandra 0.1.0 83.1, olmOCR-2 82.4, PaddleOCR-VL 80.0.

### 1.2 The parsers

**Open, sub-2B VLM parsers** are the 2026 sweet spot: PaddleOCR-VL-1.6
([arXiv 2606.03264](https://arxiv.org/abs/2606.03264)), MinerU2.5-Pro, GLM-OCR
([arXiv 2603.10910](https://arxiv.org/abs/2603.10910)), dots.ocr
([arXiv 2512.02498](https://arxiv.org/abs/2512.02498)), DeepSeek-OCR-2
([arXiv 2601.20552](https://arxiv.org/abs/2601.20552); its value is visual-token
compression, not top accuracy), olmOCR-2 (7B; "<$200 per million pages"
self-hosted, [README](https://github.com/allenai/olmocr)), Nanonets-OCR2-3B
(emits HTML tables, image descriptions, checkbox and signature tags),
Granite-Docling-258M (table TEDS-structure 0.97 at 258M parameters,
[HF card](https://huggingface.co/ibm-granite/granite-docling-258M)).

**Classical layout pipelines** still have a role. Docling (IBM, MIT licence)
is the most complete *toolkit*: a lossless `DoclingDocument` model, DocTags,
hierarchical chunkers, and a VLM pipeline that runs Granite-Docling. Its raw
OCR score trails the VLMs (50.3 on olmOCR-bench in Datalab's run **[vendor,
competitor]**); its strength is structure fidelity. Marker 2 (Datalab, Jul 2026)
reports 76.0 balanced / 23.7 pages/s no-OCR on CPU **[vendor]**, which is the
speed that matters for the born-digital majority of a corpus.

**Commercial APIs**, price per page where the vendor publishes it:

| Service | Price | Note |
|---|---|---|
| Mistral OCR 4 (Jun 2026) | $4 / 1k pages, $2 batch | bboxes, block classes, per-word confidence, self-hostable; 85.20 olmOCR-bench **[vendor]** |
| Mistral OCR 3 | $2 / 1k, $1 batch | |
| LlamaParse | $0.00125–$0.056 / page by tier | ParseBench (LlamaIndex-authored) tops at 84.9% and says no method is strong on all five dimensions |
| Unstructured | $0.015 / page flat | |
| Gemini native PDF | 258 input tokens / page fixed | ≈ $0.0003 (Flash-Lite) to ≈ $0.009 (Pro) per page including ~700 output tokens; output dominates |
| Claude native PDF | 1,500–3,000 text tokens/page + image tokens; ≤ 600 pages/request | |
| Azure Document Intelligence Layout | $10 / 1k pages **[unverified]** | |

Frontier page-as-image parsing is 5–20× the cost of Mistral OCR batch and
50–100× the marginal cost of self-hosted olmOCR.

### 1.3 OCR accuracy is not retrieval accuracy

**InduOCRBench** ([ACL Industry 2026](https://aclanthology.org/2026.acl-industry.60/)):
"high OCR accuracy does not necessarily translate into strong downstream RAG
performance … structural and semantic errors can cause substantial retrieval
failures even when WER/CER remains low." OHR-Bench (ICCV 2025) reports even the
best OCR loses ~5 F1 end-to-end versus ground truth **[unverified: secondary]**.
Table structure is now a small-model-solvable problem (TEDS 93–97); reading
order on photographed pages is not (best edit distance ≈ 0.11–0.13).

**Implications for `parse`.**

- **Swap.** Default to a sub-2B open VLM parser behind a lossless document
  model; keep a text-layer fast path for born-digital PDFs; escalate
  low-confidence pages to a commercial VLM. This is exactly the per-document
  routing the `Parser.can_parse` contract was designed for; the parsers change,
  the seam does not.
- **Knob.** A confidence threshold for escalation, next to the existing
  `min_reading_order_confidence`.
- **Evaluate on retrieval, not CER.** Score a parser by running the golden set
  through it (`RealDocBench`/`InduOCRBench` style), which the frame's
  span-anchored gold already permits: a parser change is an ablation arm.

---

## 2. Segment

### 2.1 The foundational results still stand

- **Chroma, *Evaluating Chunking Strategies for Retrieval*** (Jul 2024): token-level
  recall/precision/IoU. Reducing overlap *improves* IoU; the OpenAI default of
  800/400 overlap had "the lowest scores across all other metrics".
- **Qu, Tu & Bao, *Is Semantic Chunking Worth the Computational Cost?***
  ([arXiv 2410.13070](https://arxiv.org/abs/2410.13070), NAACL 2025 Findings):
  fixed-size chunking "consistently outperformed semantic chunking" on realistic
  documents; semantic wins only on artificially stitched ones.
- **Late chunking** (Jina, [arXiv 2409.04701](https://arxiv.org/abs/2409.04701)):
  embed the whole document, pool per chunk. nDCG@10 gains grow with document
  length (NFCorpus 23.5 → 30.0; SciFact 64.2 → 66.1; Quora unchanged).
- **Dense X / propositions** (EMNLP 2024) and **RAPTOR** (ICLR 2024) both showed
  gains in 2024; see §2.3 for how they fared in 2026 comparisons.

### 2.2 2025

- **NVIDIA, *Finding the Best Chunking Strategy*** (Jun 2025): across five datasets,
  **page-level chunking was best on average (0.648) with the lowest variance**;
  FinanceBench preferred 1,024 tokens, earnings calls 512.
- **Rethinking Chunk Size for Long-Document Retrieval** ([arXiv 2505.21700](https://arxiv.org/abs/2505.21700)):
  64–128 tokens for fact lookups, 512–1,024 for context-heavy questions; the
  optimum differs per embedding model.
- **ConTEB / InSeNT** (Illuin, [arXiv 2505.24782](https://arxiv.org/abs/2505.24782)):
  late chunking alone +9.0 nDCG@10 over standard embedding; post-training +23.6;
  applied to late-interaction models *untrained* it gives −0.3.
- **RAGSmith** ([arXiv 2511.01386](https://arxiv.org/abs/2511.01386)): 46,080 pipeline
  configurations over six domains; expansion, reranking and augmentation choices
  are domain-dependent; passage compression was never selected.

### 2.3 2026 systematic comparisons

Four independent studies, four consistent conclusions:

- **Bennani & Moslonka** ([arXiv 2601.14123](https://arxiv.org/abs/2601.14123)):
  **"overlap provides no measurable benefit and increases indexing cost"**; a
  quality cliff beyond ~2.5k tokens; sentence chunking matches semantic up to
  ~5k tokens at lower cost.
- **Shaukat, Adnan & Kuhn** ([arXiv 2603.06976](https://arxiv.org/abs/2603.06976)):
  36 methods × 5 embedders × 6 domains. **Paragraph-group chunking best**
  (nDCG@5 ≈ 0.459); naive fixed-character worst (nDCG@5 < 0.244); larger
  embedders remain vulnerable to bad chunking.
- **Śmigielski et al.** ([arXiv 2606.00881](https://arxiv.org/abs/2606.00881)):
  8 methods × 10 datasets. **DenseX/propositions worst on recall (27.4%) and
  slowest (15 h average)**; fixed-size under 1 s; "more computationally
  expensive chunking methods do not yield meaningful effectiveness improvements".
- **Adaptive Chunking** (LREC 2026, [arXiv 2603.25333](https://arxiv.org/abs/2603.25333)):
  per-document chunker selection from five intrinsic metrics lifts answer
  correctness 62–64% → 72% without changing models.

### 2.4 Tables

**STC, Structure-Aware Tabular Chunking** ([arXiv 2605.00318](https://arxiv.org/abs/2605.00318),
May 2026): row-tree, each row as a key–value block, token-capped splits on
structural boundaries, overlap-free merging. On MAUD it cut chunk count 40%
versus recursive splitting and raised hybrid MRR 0.358 → 0.595, BM25 Recall@1
0.366 → 0.754. **TableRAG** (EMNLP 2025, [arXiv 2506.10380](https://arxiv.org/abs/2506.10380))
shows that flattening "disrupts the intrinsic tabular structure" for multi-hop
and aggregation questions and keeps a SQL-queryable grid instead.

### 2.5 Contextualised embeddings as a segment/index seam

Two routes now exist for giving a chunk its document context *without* an LLM
summary: late chunking (§2.1) and document-batched embedders that return one
vector per chunk (Voyage `voyage-context-4`, Jun 2026: +2.08% NDCG@10 chunk-level
over its predecessor, +7.11% over single-vector on LongEmbed **[vendor]**). Both
couple `segment` to `index`: the embedder must see the whole document, so the
`Index.upsert` batch has to arrive grouped by document, which the enrich stage
already guarantees.

**Implications for `segment`.**

- **Confirmed defaults.** Structure-first boundaries, `overlap_tokens: 0`, table
  row groups with repeated headers. These were design bets in 2024; they are
  the measured winners in 2026.
- **Knob.** `max_tokens` should be set by query mix, not by habit: 256–512 for
  factoid corpora, 1,024 for analytical ones, and never past ~2.5k. Page-level
  units are the right fallback for structure-poor documents; the
  `whole_document` segmenter is already the arm to test that on short files.
- **Not worth a default.** Semantic, LLM-driven and proposition chunkers. Keep
  them as optional swaps; the 2026 evidence puts them at the wrong end of the
  cost/benefit curve.
- **Capability.** A document-batched embedder interface (late chunking or
  contextual embeddings) is the one genuinely new seam; it belongs in the dense
  `Index` implementation, not in a new stage.

---

## 3. Enrich

### 3.1 LLM-written chunk context

The Anthropic figures (5.7% → 3.7% → 2.9% → 1.9% top-20 failure; ≈ $1.02 per
million document tokens with prompt caching) are accurate as published. What
2025–26 adds:

- **Merola & Singh** ([arXiv 2504.19754](https://arxiv.org/abs/2504.19754), ECIR 2025
  workshop), NFCorpus subsample with Jina-v3: fixed-window nDCG@10 0.291,
  contextual retrieval + rank fusion 0.308 (+5.8%), late chunking 0.294.
  Contextualisation needed ~20 GB VRAM, forcing the subsample. "Both cannot be
  considered definitive solutions."
- **Zhou, Wang, Koopman & Zuccon, *Beyond Chunk-Then-Embed*** ([arXiv 2602.16974](https://arxiv.org/abs/2602.16974),
  Feb 2026): contextualised chunking "improves in-corpus effectiveness but
  degrades in-document retrieval", and simple structure-based segmentation beats
  LLM-guided segmentation for in-corpus retrieval.

No independent reproduction of the exact 5.7 → 1.9 ladder was found. The
effect is real; its size is a property of the corpus and the query mix, which is
precisely why the frame measures it rather than assuming it.

### 3.2 Other enrichers with measured upside

- **Hypothetical prompt embeddings (HyPE)** ([arXiv 2607.29402](https://arxiv.org/abs/2607.29402),
  IEEE Access 2025): pre-generate questions per chunk, embed *those*, map back
  to the chunk. Up to +42 pp context precision and +45 pp claim recall on six
  datasets; zero query-time cost; unlike doc2query the expansions never enter
  the text index, so they cannot bloat it. Fits the `Enrichment.extra` slot plus
  a dense index that accepts extra vectors per unit.
- **Section-path headers** (the frame's `section_prefix`): the dsRAG project
  reports FinanceBench 32% → 96.6% with AutoContext + RSE **[vendor]**, header
  only 6.04/10 versus 8.42 with full context. No independent replication; the
  control-arm role in this frame stands regardless.
- **Field extraction into a structured store.** Evidence is indirect but
  consistent: TableRAG's SQL executor, RealDocBench's field-level scoring, and
  LlamaParse pricing extraction as a separate product. No clean ablation of
  "extract fields at ingestion versus not" was found; the frame's
  `by_query_type` slice is how one would produce it.
- **Figure and table captions.** Parsers now emit them natively (Nanonets-OCR2,
  Mistral OCR 4, Docling picture description). A quantified ablation showing
  captions alone help *text* retrieval was only found in secondary sources
  **[unverified]**; ViDoRe v3 (§4.3) suggests a visual index is the stronger
  answer where figures carry the evidence.

### 3.3 Graph enrichment

The strongest 2026 evidence on graph-based RAG comes from benchmarks built to
test it fairly:

- **GraphRAG-Bench / *When to use Graphs in RAG*** ([ICLR 2026, arXiv 2506.05690](https://arxiv.org/abs/2506.05690)):
  on fact retrieval, basic RAG + rerank 60.92% versus HippoRAG2 60.14,
  LightRAG 58.62, Microsoft GraphRAG 49.29. On complex reasoning HippoRAG2
  53.38 versus basic RAG 42.93. Per-query context: MS-GraphRAG global ~331k
  tokens, LightRAG ~101k, HippoRAG2 ~1,020, vanilla ~879. GraphRAG "frequently
  underperforms vanilla RAG on many real-world tasks".
- **Zeng et al.** ([arXiv 2506.06331](https://arxiv.org/abs/2506.06331), rev. Aug 2026):
  after removing unrelated test questions and assessment biases, GraphRAG's
  gains are "much more moderate than reported previously".
- **WildGraphBench** ([arXiv 2602.02053](https://arxiv.org/abs/2602.02053)): graphs help
  multi-fact aggregation from a moderate number of sources, and hurt
  fine-grained summarisation.
- **LazyGraphRAG** (Microsoft, Nov 2024): indexing cost equal to vector RAG,
  0.1% of full GraphRAG; open-source release status still unclear in 2026
  **[unverified]**. **EraRAG** ([arXiv 2506.20963](https://arxiv.org/abs/2506.20963))
  addresses incremental graph updates.

**Implications for `enrich`.**

- **Confirmed.** The enricher chain with the section-prefix control arm is the
  right shape: the 2026 literature's main complaint about contextualisation
  papers is exactly that they lack the "does any prefix help?" control.
- **Knob.** Contextualisation on by default only for corpora where the
  ~$1/M-token cost is negligible, and *measured* per corpus because it can hurt
  within-document questions.
- **Capability.** A HyPE-style enricher writing extra vectors per unit. A graph
  enricher is optional and should be lazy or incremental, gated on a query mix
  heavy in multi-hop or sensemaking questions; it is not a fact-retrieval tool.

---

## 4. Index

### 4.1 Dense embeddings

The MTEB(Multilingual) v2 top has changed three times in twelve months. Verified
against model cards and papers:

| Model | Params | Context | Dims / Matryoshka | Licence | Source |
|---|---|---|---|---|---|
| Qwen3-Embedding 0.6B / 4B / 8B | 0.6–8B | 32K | 1024 / 2560 / 4096, MRL | Apache-2.0 | MMTEB 64.33 / 69.45 / 70.58 ([arXiv 2506.05176](https://arxiv.org/abs/2506.05176)) |
| Gemini Embedding 2 | — | 8K | 128–3072 MRL; text, image, video, audio, PDF | API | MTEB(Multilingual) 69.9 in its own paper ([arXiv 2605.27295](https://arxiv.org/abs/2605.27295)) |
| Voyage 4 family (large / 4 / lite / nano open-weights) | — | 32K | 2048–256 MRL; int8, binary; *shared space across sizes* | API + Apache-2.0 nano | [Voyage, Jan 2026](https://blog.voyageai.com/2026/01/15/voyage-4/) |
| voyage-context-4 | — | >32K, auto-chunking | 2048–256 MRL | API | [Voyage, Jun 2026](https://blog.voyageai.com/2026/06/29/voyage-context-4/) **[vendor]** |
| Cohere Embed v4 | — | 128K | 256–1536 MRL; text + image | API | only 128K-context option; 8–11% behind Voyage 4 on Voyage's eval **[vendor]** |
| jina-embeddings-v4 | 3.8B | 32K | 2048 single + 128/token multi-vector | Qwen research licence | [arXiv 2506.18902](https://arxiv.org/abs/2506.18902) |
| jina-embeddings-v5-text small / nano | 677M / 239M | 32K / 8K | robust to truncation and binary | CC-BY-NC-SA | [arXiv 2602.15547](https://arxiv.org/abs/2602.15547) |
| EmbeddingGemma | 308M | 2K | 768 → 128 MRL | Gemma | "#1 open < 500M" ([arXiv 2509.20354](https://arxiv.org/pdf/2509.20354)) |
| Snowflake arctic-embed-l-v2.0 | 303M | 8K | 1024, MRL, QAT to 128 bytes | Apache-2.0 | [arXiv 2412.04506](https://arxiv.org/abs/2412.04506) |
| BGE-M3 | 568M | 8K | dense + sparse + multi-vector | MIT | no successor; BAAI's 2025 effort went into BGE-Reasoner |
| OpenAI text-embedding-3-large | — | 8K | 3072 MRL | API | no successor announced as of Sep 2026 |

Ranking claims are date- and version-specific: Gemini Embedding 2's own 69.9 is
*below* Qwen3-8B's 70.58, whatever the launch blogs said; a Tencent KaLM 12B at
72.32 appears only in secondary sources **[unverified]**.

**Implications.** `INDEX` **swap**: Qwen3-Embedding-4B is the open-weights value
point for chunk retrieval at 32K context; 0.6B for CPU. **Knob**: expose the
Matryoshka dimension and the instruction/task prefix (Qwen3, Gemini and Voyage
are instruction-aware). The `configs/full.yaml` note that a non-Matryoshka model
costs a full re-index to undo is now doubly true: Voyage 4's shared embedding
space and Drift-Adapter (§4.5) make model migration cheap *only* if the seam
was left open.

### 4.2 Learned sparse and BM25

- **SPLADE-v3** ([arXiv 2403.06789](https://arxiv.org/abs/2403.06789)) is still the
  reference open learned-sparse model; the TREC 2025 RAG baseline used it as one
  of two first-stage legs.
- **OpenSearch neural sparse v3** ([blog, Sep 2025](https://opensearch.org/blog/advancing-search-with-opensearch-v3-neural-sparse-models-and-a-multilingual-retrieval-model/)):
  inference-free document-side models; BEIR nDCG@10 v3-gte 0.546 versus
  v2-distill 0.528 and v3-distill 0.517. The distill variant is *lower* than
  its predecessor; the gte variant is the real gain.
- **Elastic ELSER v2**: +18% average nDCG@10 over BM25 on BEIR; ELSER v3 announced,
  no numbers yet.
- **HAKARI-Bench** ([arXiv 2606.22778](https://arxiv.org/abs/2606.22778), Jun 2026):
  for SPLADE-v3, query-side pruning 32 → 8 active terms costs 2.5–3.6 points;
  99% quality needs ≥ 24 query terms and ≥ 128 document terms.
- **BM25** remains the always-on lexical leg. BM25S (up to 500× faster than
  rank_bm25) is the practical Python implementation; on BRIGHT, query-side BM25
  weighting (BM25Q) beats standard BM25 for long narrative queries (14.8 vs 13.7,
  [arXiv 2509.02558](https://arxiv.org/abs/2509.02558)).

**Implication.** `INDEX` **capability**: a learned-sparse leg as a third index
kind. It is nearly free at query time with inference-free models, but the
weakest-link finding (§6) means it must earn its place on the ablation table.

### 4.3 Late interaction and visual document retrieval

- **ViDoRe V3** ([arXiv 2601.08620](https://arxiv.org/abs/2601.08620), Jan 2026; 26K
  pages, 3,099 human-verified queries, six languages): "for a given parameter
  count, visual retrievers outperform textual retrievers, and late interaction
  methods score higher than dense methods". ColEmbed-3B-v2 (visual) 59.8
  nDCG@10 versus Qwen3-8B text 51.0. Adding a *text* reranker to jina-v4 text
  retrieval moves it 50.4 → 63.6; a *visual* reranker adds 0.2. Hybrid top-5
  visual + top-5 text beat either modality alone on hard queries.
- **Nemotron ColEmbed V2** ([arXiv 2602.03992](https://arxiv.org/abs/2602.03992)):
  8B tops ViDoRe V3 at 63.42. Storage for 1M pages at fp16: **3.8 GB
  single-vector versus 8,789 GB late-interaction**; truncating to 128 dims keeps
  95% of accuracy at 3% of the storage.
- **Small models**: ColModernVBERT (250M) is ~0.6 nDCG@5 below ColPali and runs on
  CPU ([arXiv 2510.01149](https://arxiv.org/abs/2510.01149)).
- **Counter-evidence**: *Lost in OCR Translation?* ([arXiv 2505.05666](https://arxiv.org/abs/2505.05666))
  finds OCR-text RAG generalises better to unseen document quality than ColPali;
  *Document-as-Image Representations Fall Short for Scientific Retrieval*
  ([arXiv 2604.18508](https://arxiv.org/abs/2604.18508)) finds text wins on long
  born-digital scientific papers even for figure queries.
- **Making multi-vector tractable**: MUVERA fixed-dimensional encodings
  ([arXiv 2405.19504](https://arxiv.org/abs/2405.19504)) let any single-vector ANN
  serve multi-vector search; Qdrant measures MUVERA + MaxSim rescoring at 0.343
  versus full MaxSim 0.347 nDCG at ~7× lower latency. Vespa's binarised ColPali
  cuts 16 KB/page to 2 KB with a ~1-point loss after float rescoring. ColChunk /
  visual late chunking ([arXiv 2604.10167](https://arxiv.org/abs/2604.10167)):
  > 90% storage reduction, +9 nDCG@5 over single-vector across 24 datasets.

**Implication.** The `visual` index kind in `configs/full.yaml` was the right
seam. Two additions: it should be *routed* per page (born-digital prose gets
text embeddings; table-, figure- and scan-heavy pages get page-image
embeddings, or both), and its results should be fused and then reranked by a
*text* reranker, which is where ViDoRe v3 finds the largest single gain. Budget
multi-vector storage explicitly (pooling, 128-dim truncation, binarisation, or
MUVERA).

### 4.4 Tree and file-system indexes

- **PageIndex** (Vectify): layout-derived table-of-contents tree, LLM node
  summaries with page ranges, retrieval by LLM navigation. Vendor figures:
  FinanceBench 98.7% (Mafin 2.5), indexing ≈ $0.001/page, native-PDF-in-context
  costing 2.1× (52 pages) to 16.6× (420 pages) more per query than tree
  navigation **[vendor]**; the vendor itself notes FinanceBench label errors and
  its single-document scope. The "~50% for vector RAG" comparator is
  **[unverified]**.
- **KohakuRAG** ([arXiv 2603.07612](https://arxiv.org/abs/2603.07612)): four-level tree
  with bottom-up embedding aggregation; won WattBot 2025; hierarchical dense ≈
  hybrid, BM25 adding only 3.1 pp. **BookRAG** ([arXiv 2512.03413](https://arxiv.org/abs/2512.03413))
  combines a TOC tree with an entity graph. **DTCRS** ([arXiv 2604.07012](https://arxiv.org/abs/2604.07012))
  builds RAPTOR-style trees only when the query type needs them.
- **Agentic file-system retrieval.** LlamaIndex (Jan 2026): a grep/glob/read agent
  beat hybrid RAG on five papers (correctness 8.4 vs 6.4) and lost at 100 and
  1,000 documents. *Is Grep All You Need?* ([arXiv 2605.15184](https://arxiv.org/abs/2605.15184)):
  grep generally more accurate than vector retrieval on LongMemEval, but the
  harness and tool presentation changed results more than the retrieval method.
  *Keyword search is all you need* ([AAAI 2026, arXiv 2602.23368](https://arxiv.org/abs/2602.23368)):
  a ReAct agent with `pdfgrep` reaches 91.48% of RAG answer correctness on
  average and beats it on FinanceBench (32.71% vs 24.24%), with documented
  degradation on large documents.

**Implication.** `INDEX` **capability**: a `tree` index kind (TOC tree with node
summaries, ~$0.001/page) that coexists with the chunk indexes, plus a
section-as-file layout of the parsed document so grep-style agents can navigate
it at zero model cost. `ROUTE`: long single-document analytical questions go
there; broad multi-document lookups stay on vectors.

### 4.5 Store internals: quantisation, filters, deletion, migration

- **Quantisation** (HAKARI-Bench, 33 models × 551 tasks): int8 costs −1.95
  nDCG@10×100 and binary −6.50; with float rescoring of the top 100, int8 −0.09
  and binary −0.93. Vespa (Jan 2026): 100M × 768-d = 307 GB fp32 / 77 int8 /
  9.6 binary; int8 is 2.7–3.4× faster on CPU but 4–5× *slower* than fp32 on GPU.
- **Filtered search**: ACORN-style traversal (default in Weaviate ≥ 1.34,
  available in Qdrant) gives 5–20× lower latency at < 5% selectivity but recall
  collapses below ~1%; RACORN-1 ([arXiv 2607.00768](https://arxiv.org/html/2607.00768))
  adds an adaptive fallback.
- **Deletion is not erasure.** *Ghost Vectors* ([arXiv 2606.18497](https://arxiv.org/abs/2606.18497),
  Jun 2026): soft-deleted vectors in three HNSW implementations remain
  recoverable from raw index files (25.5% exact recovery of person names, 100%
  of patient age/gender markers). Qdrant vacuums a segment only past 20% deleted;
  Milvus never removes vectors from an existing index and rebuilds the segment.
- **Embedding migration.** *Drift-Adapter* ([EMNLP 2025, arXiv 2509.23471](https://arxiv.org/abs/2509.23471)):
  a learned map from old to new embedding space recovers 95–99% of full
  re-embedding recall at < 10 µs query overhead. Voyage 4's shared space across
  model sizes lets documents indexed with one model be queried with another.
  Practitioner reports **[secondary]**: throughput quotas, not price, bottleneck
  a 40B-token re-embed; one legal deployment lost 11% nDCG@10 moving to a model
  4 points *higher* on MTEB.

**Implications.** `INDEX` **knobs**: `quantization`, `rescore_top_k` (always on
when quantised), `mrl_dim`, `filter_strategy`. The frame's `Index.delete`
contract ("must actually remove") is stronger than what most stores do by
default; the manifest should record the store's compaction policy, and a
compaction or key-rotation step belongs after any bulk re-index. Model
migration needs a 200–500-query recall gate, which is what the eval harness is
for.

---

## 5. Route

### 5.1 Cheap routers work

- **RAGRouter-Bench** ([arXiv 2604.03455](https://arxiv.org/abs/2604.03455), Apr 2026;
  7,727 queries, four domains): a TF-IDF + SVM complexity router reaches
  macro-F1 0.928 and saves 28.1% of tokens versus always using the most
  expensive strategy; lexical features beat MiniLM embeddings by 3.1 F1.
- **Adaptive-RAG** (NAACL 2024) remains the reference for routing among
  no-retrieval, single-step and multi-step.
- Retrieval scores cannot be used for routing without calibration: Abdallah et
  al. ([SIGIR 2026, arXiv 2604.03676](https://arxiv.org/abs/2604.03676)) find
  "confidence calibration is consistently weak" across 14 retrievers.

This vindicates the frame's choice of a rules router as the reference and
control arm.

### 5.2 Query rewriting and decomposition

- ***Better Together*** ([arXiv 2609.05637](https://arxiv.org/abs/2609.05637), Sep 2026):
  against a strong baseline (BGE + cross-encoder + MMR), any *single* rewrite
  including HyDE is "at best competitive"; a *union* of 4–5 rewrites gives
  +12.5 to +13.8 HIT@10 on enterprise data and −2.4 on AmbigNQ; a
  confidence-gated router keeps ~50% of the gain while rewriting < 40% of
  queries.
- ***When Should Queries Be Decomposed?*** ([EMNLP Findings 2026, arXiv 2606.08577](https://arxiv.org/abs/2606.08577)):
  decomposition at first-stage retrieval **hurts** (semantic dilution);
  decomposition at *rerank* helps (constraint verification).
- Akarsu et al. ([arXiv 2604.01733](https://arxiv.org/abs/2604.01733)): on 23,088
  financial text-and-table queries, HyDE (R@5 0.544) and multi-query + RRF
  (0.640) both *lose* to plain hybrid RRF (0.695); CRAG 0.658.

**Implication.** The `IterativeRetriever` currently issues the router's
sub-queries as first-stage rounds. The 2026 evidence says: issue the *union of
rewrites* as parallel first-stage queries and fuse, but pass *decompositions*
to the reranker rather than the retriever. That is a change inside the
retriever and reranker implementations, not to the pipeline shape.

### 5.3 Long context versus retrieval

- **Self-Route** ([EMNLP 2024 Industry, arXiv 2407.16833](https://arxiv.org/abs/2407.16833)):
  long context beats RAG when the corpus fits; routing on self-reflection cuts
  cost 39–65% within ~0.2–2.2% of long-context quality.
- **Databricks** ([arXiv 2411.03538](https://arxiv.org/abs/2411.03538)): "only a
  handful of the most recent state of the art LLMs can maintain consistent
  accuracy at long context above 64k tokens".
- **NoLiMa** ([ICML 2025, arXiv 2502.05167](https://arxiv.org/abs/2502.05167)): remove
  lexical overlap between needle and question and 11 of 13 models claiming
  ≥ 128K drop below 50% of their short-context score at 32K (GPT-4o 99.3% →
  69.7%).
- **Context Rot** (Chroma, Jul 2025; 18 models): focused ~300-token prompts beat
  ~113k full prompts for every model on LongMemEval.
- Anthropic's own guidance: below ~200k tokens (~500 pages), skip retrieval and
  put the corpus in the prompt. Prompt caching lowers the price of a static
  prefix; it does not fix context rot, and per-query cost still scales with
  corpus size.

No 2026 primary study re-runs the crossover with 1M-token models on document
corpora; the 2026 material is blog-level **[unverified]**.

### 5.4 Agentic retrieval

- Anthropic, *Effective context engineering for AI agents* (Sep 2025): just-in-time
  loading via identifiers; Claude Code uses glob/grep at runtime; "runtime
  exploration is slower than retrieving pre-computed data … the most effective
  agents might employ a hybrid strategy". Anthropic's multi-agent research
  system uses ~4× (single agent) and ~15× (multi-agent) the tokens of chat.
- **Search-R1** ([arXiv 2503.09516](https://arxiv.org/abs/2503.09516)): RL-trained
  multi-turn search improves over RAG baselines by 41% (7B) and 20% (3B) on
  seven QA sets.
- ***Do We Still Need GraphRAG?*** ([arXiv 2604.09666](https://arxiv.org/abs/2604.09666)):
  multi-round agentic search over dense RAG narrows the gap to GraphRAG on
  multi-hop, at higher query cost.
- **Compute allocation** ([arXiv 2603.14635](https://arxiv.org/abs/2603.14635), Mar 2026,
  BRIGHT): reranking gains +21% from k=10 to k=100; query expansion gains only
  +1.1 from a weak to a strong LLM; inference-time "thinking" ≈ no gain at either
  stage. Concentrate compute on reranking.

**Implications for `route`.** **Confirmed**: the three-path shape with a
budgeted `ITERATIVE` path and a rules router as control. **Capability**: expose
the query pipeline as *tools* (lexical, dense, structured, page-image, tree)
callable by an agent loop under `step_budget`, rather than only as one fused
pass; `RetrievalResponse` already carries what such a loop needs. **Knob**: a
long-context branch ("corpus under N tokens → no index") and a confidence gate
for multi-rewrite.

---

## 6. Retrieve and fuse

- **Hybrid still wins with 2026 embedders.** Vespa (Jan 2026), across every
  < 500M open embedder tested: BM25 + vector improves 3–5 nDCG points over
  vector alone under RRF, atan or linear fusion. Akarsu et al. (Apr 2026):
  BM25 R@5 0.644 *beats* text-embedding-3-large 0.587; hybrid RRF 0.695; hybrid
  + Cohere Rerank v4 Pro 0.816. T²-RAGBench (EACL 2026) finds "Hybrid BM25" best
  with text-embedding-3-large alone at 33.8% R@1. TREC 2025's best retrieval run
  fused SPLADE-v3 and Arctic-Embed-L via RRF, then reranked.
- **Weakest link.** *Balancing the Blend* ([arXiv 2508.01405](https://arxiv.org/abs/2508.01405);
  11 datasets, four retrieval paths): one poor leg drags fusion down; there is
  no universal configuration. *Drowning in Documents* ([arXiv 2411.11767](https://arxiv.org/abs/2411.11767)):
  reranker effectiveness "gradually declines" and can worsen results as depth
  grows. FreshStack (NeurIPS 2025): reranking failed to help on two of five
  topics.
- **RRF versus convex combination.** Bruch, Gai & Ingber ([ACM TOIS 2023, arXiv 2210.11934](https://arxiv.org/abs/2210.11934)):
  RRF is sensitive to its parameter; a convex combination of normalised scores
  beats it in- and out-of-domain and its single weight tunes with a handful of
  labelled queries. No stronger 2025–26 result on fusion functions was found;
  small-k RRF (10–20) and weighted RRF have practitioner support only
  **[unverified]**.

**Implications.** **Confirmed**: fan out over both legs, fuse, then rerank; the
`dense-only` and `lexical-only` arms are the right way to check. **Knob**: the
RRF `k` is exposed but 60 is a prior, not a finding; add an arm at k=20. A
convex fuser with min-max normalisation and a fitted α is a legitimate **swap**
the `Fuser` contract already allows ("must state its normalisation"). Add a
leg-quality gate: a leg whose standalone nDCG is far below the others is dropped
from fusion, per the weakest-link result.

---

## 7. Rerank

### 7.1 Pointwise rerankers

| Model | Params | Context | BEIR nDCG@10 | Licence / price | Source |
|---|---|---|---|---|---|
| bge-reranker-v2-m3 | 568M | 8K | 53.94 | MIT | Mixedbread's run |
| mxbai-rerank-large-v2 | 1.5B | 8K | 57.49 (Cohere 3.5 = 55.39 in same table) | Apache-2.0 | [Mixedbread](https://www.mixedbread.com/blog/mxbai-rerank-v2) |
| Qwen3-Reranker 0.6B / 4B / 8B | 0.6–8B | 32K | MTEB-R 65.80 / 69.76 / 69.02 | Apache-2.0 | [arXiv 2506.05176](https://arxiv.org/html/2506.05176) |
| jina-reranker-v3 / v3.5 | 0.6B | — | 61.94 / 63.20 | non-commercial | [arXiv 2509.25085](https://arxiv.org/abs/2509.25085), [arXiv 2607.18152](https://arxiv.org/abs/2607.18152) |
| Voyage rerank-2.5 | API | 32K | +7.94% vs Cohere 3.5, +2.25% vs Qwen3-Reranker-8B **[vendor]** | $0.05 / M tokens | [Voyage, Aug 2025](https://blog.voyageai.com/2025/08/11/rerank-2-5/) |
| Cohere Rerank 4 Fast / Pro | API | 32K | no public per-dataset numbers | — | Rerank 3.5 deprecated Jul 2026 |

Qwen3-Reranker-4B matches or beats the 8B on several sets and is the open
value point. HAKARI-Bench: Qwen3-Reranker-0.6B macro 68.03 versus the best dense
retriever 64.93; multilingual cross-encoders "collapse on long queries".

### 7.2 Listwise and reasoning rerankers

- **ReasonRank** ([ACL 2026, arXiv 2508.07050](https://arxiv.org/abs/2508.07050)):
  BRIGHT nDCG@10 7B 35.74 / 32B 38.03 versus Rank1-7B 27.23, RankZephyr 22.64.
  **On seven BEIR datasets ReasonRank-7B 54.35 versus RankZephyr 54.14: no gain
  on ordinary queries.** Listwise is 2–2.7× faster than pointwise Rank1.
- ***How good are LLM-based rerankers?*** ([EMNLP Findings 2025, arXiv 2508.16757](https://arxiv.org/abs/2508.16757);
  22 methods): LLM rerankers shine on familiar queries and degrade on novel
  ones; small cross-encoders stay cost-competitive.
- **FIRST** (EMNLP 2024): single-token listwise scoring, 21–42% faster than
  full-generation listwise reranking.
- TREC 2025 RAG baseline: RankQwen3-32B over top-1000 → top-100.

### 7.3 Depth

The 2026 compute-allocation study puts the reranking gain at +21% from k=10 to
k=100 and flat thereafter; *Drowning in Documents* shows the decline past the
knee. Practitioner curves converge on 50–100 candidates for pointwise
rerankers **[unverified]**.

**Implications for `rerank`.** **Swap**: Qwen3-Reranker-4B or jina-reranker-v3.5
open; Voyage rerank-2.5 or Cohere Rerank 4 managed. **Knob**: `input_top_k` 100
as the default for pointwise rerankers on LOOKUP, 50 where latency is tight;
listwise windows of 10–20 with stride 5–10. **Reference guidance, restated**:
small pointwise on LOOKUP; listwise or reasoning rerankers only on ITERATIVE and
only over the top 20–30. The per-path `rerank` key in `PathSpec` is exactly the
place to express that.

---

## 8. Evaluate

- **TREC 2025 RAG track** ([arXiv 2603.09891](https://arxiv.org/abs/2603.09891)): 150+
  submissions; an LLM-judge ensemble "approximates manual judgments at the run
  level" with "greater variability" per topic; participant LLM-judge runs
  reached only κ ≈ 0.1–0.2 with humans.
- **UMBRELA** ([arXiv 2406.06519](https://arxiv.org/abs/2406.06519)): GPT-4o versus
  human on TREC DL, Cohen's κ 0.31–0.37 at label level, Kendall τ 0.87–0.94 on
  system rankings. **Clarke & Dietz** ([arXiv 2412.17156](https://arxiv.org/abs/2412.17156),
  rev. Jan 2026) show systems that exploit LLM judges and a circularity when the
  judge model also reranks.
- **BRIGHT** ([arXiv 2407.12883](https://arxiv.org/abs/2407.12883)): SFR-Embedding-Mistral
  scores 59.0 on MTEB and 18.3 nDCG@10 on BRIGHT. **FreshStack** (NeurIPS 2025):
  nugget-level judgments; off-the-shelf retrievers "significantly underperform
  oracle approaches on all five topics". MTEB rank does not predict in-domain
  retrieval quality.
- **RAGBench** ([arXiv 2407.11005](https://arxiv.org/abs/2407.11005)): a fine-tuned
  RoBERTa evaluator beat LLM-based judges on its TRACe metrics.
- **Golden-set size.** No 2026 guidance specific to RAG was found; the IR
  literature (Sakai, IRJ 2015 and SIGIR 2016) sizes topic sets from desired
  paired-test power using a past topic-by-run variance matrix; classic
  collections use 50–100 topics. Practitioner consensus **[secondary]**: 50–100
  curated queries for a smoke set, 200–500 for a migration decision, paired
  bootstrap with 10k resamples for confidence intervals.

**Implications for `eval`.** **Confirmed**: span-anchored gold, the retrieval
failure rate reported beside recall, exact judges for numbers and dates, LLM
judges with their fingerprint recorded. **To add**: paired-bootstrap confidence
intervals on every delta in the ablation table (a delta without an interval is
an anecdote at n=300), and a human-verified subset reported separately, which
`GoldOrigin` already makes possible.

---

## 9. Consensus versus contested

**Consensus (two or more independent primary sources agree)**

1. Retrieval, not generation, is the dominant failure mode; oracle-vs-retrieved
   gaps of 7–30 points recur.
2. Hybrid lexical + dense, fused, then reranked at bounded depth is the best
   general single-pass default; BM25 stays competitive on numeric- and
   entity-heavy corpora.
3. Reranking is where marginal compute pays most; the gain flattens near k=100.
4. Overlap does not pay. Structure-aligned chunks (page, paragraph group,
   section, table rows) win or tie at near-zero cost; semantic, LLM and
   proposition chunkers rarely earn their cost.
5. Chunk size is task- and embedder-dependent, with a cliff past ~2.5k tokens.
6. Giving chunks document context helps, by LLM prefix, late chunking or headers.
7. Tables must keep their structure; flattening hurts aggregation questions.
8. Sub-2B VLM parsers beat classical layout pipelines on clean pages; OCR
   accuracy does not predict retrieval accuracy.
9. Vision embeddings beat OCR-text retrieval on visually rich in-distribution
   corpora; a *text* reranker on top is the largest single gain.
10. Effective context is far shorter than advertised; focused context beats
    full context even for frontier models.
11. LLM judges rank systems reliably and label items unreliably.
12. int8 + float rescoring is effectively lossless; binary + rescoring costs
    ≈ 1 point for 32× less storage.
13. Soft deletion in ANN indexes is not erasure.

**Contested or thinly evidenced**

1. Contextual retrieval versus late chunking: effectiveness versus indexing cost,
   and the in-document downside.
2. Vectorless tree navigation versus vector retrieval: vendor-run,
   single-document evidence only.
3. Grep-style agents versus vector indexes for *documents*: wins at ≤ 10
   documents and on exact-match questions, loses at 100–1,000; the harness
   confounds results.
4. Whether graphs are worth building: fair benchmarks say narrow and moderate;
   method papers say large on multi-hop.
5. Long context plus caching replacing retrieval: strong narratives, no 2026
   primary study on 1M-token models over document corpora.
6. RRF versus tuned convex combination; small-k RRF.
7. Whether a learned-sparse third leg helps once BM25 + strong dense + reranker
   are in place.
8. Reasoning rerankers' cost/benefit outside BRIGHT-style traffic.
9. Which embedding model is "#1": version- and date-specific.

---

## 10. Recommended defaults, September 2026

Two stacks, mapped to this frame's config keys. Nothing here is a recommendation
for *your* corpus; each is the starting point the ablation table should be run
against.

### Open-weights, self-hosted

| Stage | Choice | Config |
|---|---|---|
| parse | PaddleOCR-VL-1.6 / MinerU2.5-Pro / GLM-OCR behind a Docling-style lossless model; text-layer fast path for born-digital PDFs; commercial VLM escalation for low-confidence pages | `ingestion.parse.routes` on `text_layer`, `complexity`, confidence |
| segment | structural, 512–1,024 tokens by query mix, `overlap_tokens: 0`, table row groups with repeated headers, page-level fallback | `ingestion.segment` |
| enrich | `section_prefix` always; LLM contextualiser (small prompt-cached model) where cost is negligible and measured; `regex_fields` / LLM field extractor for the structured path | `ingestion.enrich.enrichers` |
| index | BM25 (BM25S) + Qwen3-Embedding-4B (MRL 1024, int8 with rescoring) + SQLite/DuckDB fields; optional learned-sparse leg; page-image index (ColModernVBERT or ColEmbed truncated to 128-d, binarised or MUVERA) for figure-heavy collections; TOC tree per long document | `ingestion.index.indexes` |
| route | rules router as control; TF-IDF/SVM complexity router as the first upgrade; confidence-gated multi-rewrite | `query.route` |
| retrieve | top-100 to top-200 per leg; filtered HNSW (ACORN) for tenant/document filters | `query.route.paths.*.top_k` |
| fuse | RRF with `k` ablated at 20 and 60, or convex α fitted on ~100 labelled queries; leg-quality gate | `query.fuse` |
| rerank | Qwen3-Reranker-4B or jina-reranker-v3.5 over top-100 → top-10/20 on LOOKUP; ReasonRank-7B listwise over top-20 on ITERATIVE only | `query.rerank`, `paths.*.rerank` |
| eval | span-anchored gold, failure rate + nDCG@10 + P@5, paired bootstrap CIs, human-verified subset | `eval` |

### Managed APIs

| Stage | Choice |
|---|---|
| parse | Mistral OCR 4 batch ($2 / 1k pages) or Gemini Flash-class page-as-image for escalation only |
| index (dense) | voyage-context-4 (document-batched, contextualised chunks) or voyage-4 with the frame's own segmenter; Gemini Embedding 2 when images and text must share a space; Cohere Embed v4 only when 128K single-call context is required |
| index (lexical, structured) | own BM25 and SQL; fusion is cheap to own |
| rerank | Voyage rerank-2.5 ($0.05 / M tokens) or Cohere Rerank 4 Fast; top-100 in, top-20 out |
| route / enrich | any frontier LLM for gated rewriting and contextualisation; skip "thinking" modes for retrieval steps (no measured gain) |

### Regardless of stack

- Below ~200k tokens of corpus for a one-shot question, do not index; put it in
  the prompt.
- Order, trim and deduplicate what goes to the reader; more context is not free
  even with caching.
- Gate every model swap on a 200–500-query recall check; MTEB rank does not
  predict your corpus.

---

## 11. What this review changed in the repo, and what to revisit next

**Changed on 2026-09-18**

- README and `ARCHITECTURE.md`: invariant 1 re-scoped to its source; invariants
  3, 4 and 5 given their 2025–26 caveats; the reranker guidance restated as
  pointwise-small versus listwise-large.
- `configs/reference.yaml`: two ablation arms added, RRF `k=20` and
  `overlap_tokens: 64`, so the two most-cited 2026 chunking and fusion findings
  are checkable on any corpus with one command.
- `configs/full.yaml`: reranker input width 100; 2026 model names in comments;
  learned-sparse and tree index kinds sketched as disabled entries.
- Docstrings in `eval/metrics.py`, `config/schema.py` and `core/stages.py` that
  quoted the invariant-1 figures as general facts now state their scope.
- `docs/ABLATION-pypi-docs.md`: the first ablation run on a real corpus, with
  the sanity-check verdicts. It caught two defects in the measurement
  apparatus (an ablation arm that changed two things while naming one, and a
  contract checker that gave a false positive on any corpus with shared
  document titles); both are fixed.

**Revisit when**

- A 2026 primary study re-runs long-context versus retrieval with 1M-token models
  over document corpora. If retrieval loses above ~500 pages, the long-context
  branch in `route` becomes the default rather than the exception.
- An independent replication of contextual retrieval on a public corpus lands.
  The sanity-check thresholds in the configs should be recalibrated to it.
- Learned-sparse inference-free models close the gap to SPLADE-v3 at zero
  query cost. Then the third leg becomes a default rather than an arm.
- ELSER v3, BGE-M3's successor, or an OpenAI embedding successor ship; §4.1 is
  the table to update.
- Fair graph-RAG benchmarks (GraphRAG-Bench, WildGraphBench) show a graph
  enricher winning on *fact* retrieval. Until then it stays optional.

---

## Sources

**Parsing and document benchmarks**
[OmniDocBench (GitHub, v1.6 leaderboard)](https://github.com/opendatalab/OmniDocBench) ·
[OmniDocBench paper](https://arxiv.org/pdf/2412.07626) ·
[MinerU2.5-Pro](https://arxiv.org/abs/2604.04771) ·
[PaddleOCR-VL-1.6](https://arxiv.org/abs/2606.03264) ·
[GLM-OCR](https://arxiv.org/abs/2603.10910) ·
[dots.ocr](https://arxiv.org/abs/2512.02498) ·
[DeepSeek-OCR-2](https://arxiv.org/abs/2601.20552) ·
[olmOCR 2](https://arxiv.org/abs/2510.19817) · [olmOCR GitHub](https://github.com/allenai/olmocr) ·
[Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) ·
[Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) ·
[Docling](https://github.com/docling-project/docling) ·
[Marker 2](https://www.datalab.to/blog/marker-2) ·
[Mistral OCR 4](https://mistral.ai/news/ocr-4/) · [Mistral OCR 3](https://mistral.ai/news/mistral-ocr-3) ·
[LlamaParse pricing](https://developers.llamaindex.ai/llamaparse/general/pricing/) · [ParseBench](https://arxiv.org/abs/2604.08538) ·
[Unstructured pricing](https://unstructured.io/pricing) ·
[Gemini document processing](https://ai.google.dev/gemini-api/docs/document-processing) ·
[Claude PDF support](https://platform.claude.com/docs/en/docs/build-with-claude/pdf-support) ·
[PureDocBench](https://arxiv.org/abs/2605.07492) ·
[Real5-OmniDocBench](https://arxiv.org/abs/2603.04205) ·
[RealDocBench](https://arxiv.org/abs/2606.07401) ·
[InduOCRBench (ACL Industry 2026)](https://aclanthology.org/2026.acl-industry.60/) ·
[OHR-Bench](https://github.com/opendatalab/OHR-Bench)

**Chunking**
[Chroma, Evaluating Chunking Strategies](https://www.trychroma.com/research/evaluating-chunking) ·
[Qu, Tu & Bao 2024](https://arxiv.org/abs/2410.13070) ·
[NVIDIA, Finding the Best Chunking Strategy](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/) ·
[Rethinking Chunk Size](https://arxiv.org/abs/2505.21700) ·
[Bennani & Moslonka 2026](https://arxiv.org/abs/2601.14123) ·
[Shaukat, Adnan & Kuhn 2026](https://arxiv.org/abs/2603.06976) ·
[Śmigielski et al. 2026](https://arxiv.org/abs/2606.00881) ·
[Adaptive Chunking (LREC 2026)](https://arxiv.org/abs/2603.25333) ·
[LumberChunker](https://arxiv.org/abs/2406.17526) · [Meta-Chunking](https://arxiv.org/abs/2410.12788) · [MoC](https://arxiv.org/abs/2503.09600) ·
[Late Chunking](https://arxiv.org/abs/2409.04701) · [Jina late-chunking results](https://github.com/jina-ai/late-chunking) ·
[ConTEB / InSeNT](https://arxiv.org/abs/2505.24782) ·
[Reconstructing Context (Merola & Singh)](https://arxiv.org/abs/2504.19754) ·
[Beyond Chunk-Then-Embed (Zhou et al.)](https://arxiv.org/abs/2602.16974) ·
[Anthropic, Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) ·
[Dense X Retrieval](https://arxiv.org/abs/2312.06648) · [RAPTOR](https://arxiv.org/abs/2401.18059) · [DTCRS](https://arxiv.org/abs/2604.07012) ·
[RAGSmith](https://arxiv.org/abs/2511.01386) ·
[dsRAG](https://github.com/D-Star-AI/dsRAG) ·
[STC, Structure-Aware Tabular Chunking](https://arxiv.org/abs/2605.00318) ·
[TableRAG (EMNLP 2025)](https://arxiv.org/abs/2506.10380) ·
[HyPE](https://arxiv.org/abs/2607.29402)

**Tree, graph and agentic indexing**
[PageIndex](https://github.com/VectifyAI/PageIndex) · [Mafin 2.5 eval](https://github.com/VectifyAI/Mafin2.5-FinanceBench) ·
[BookRAG](https://arxiv.org/abs/2512.03413) · [KohakuRAG](https://arxiv.org/abs/2603.07612) ·
[LlamaIndex, Did filesystem tools kill vector search?](https://www.llamaindex.ai/blog/did-filesystem-tools-kill-vector-search) ·
[Is Grep All You Need?](https://arxiv.org/abs/2605.15184) ·
[Keyword search is all you need (AAAI 2026)](https://arxiv.org/abs/2602.23368) ·
[Rethinking Agentic RAG](https://arxiv.org/abs/2605.27123) ·
[GraphRAG-Bench / When to use Graphs in RAG (ICLR 2026)](https://arxiv.org/abs/2506.05690) ·
[Zeng et al., Unbiased Evaluation for GraphRAG](https://arxiv.org/abs/2506.06331) ·
[WildGraphBench](https://arxiv.org/abs/2602.02053) ·
[LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) ·
[HippoRAG 2](https://arxiv.org/abs/2502.14802) · [KAG](https://arxiv.org/abs/2409.13731) · [EraRAG](https://arxiv.org/abs/2506.20963) ·
[Do We Still Need GraphRAG?](https://arxiv.org/abs/2604.09666) ·
[Search-R1](https://arxiv.org/abs/2503.09516) ·
[Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) ·
[Anthropic, Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

**Embeddings, sparse, late interaction**
[Qwen3 Embedding report](https://arxiv.org/abs/2506.05176) · [Qwen3-Embedding-8B card](https://huggingface.co/Qwen/Qwen3-Embedding-8B) ·
[Gemini Embedding 2](https://arxiv.org/abs/2605.27295) ·
[Voyage 4](https://blog.voyageai.com/2026/01/15/voyage-4/) · [voyage-context-4](https://blog.voyageai.com/2026/06/29/voyage-context-4/) · [Voyage pricing](https://docs.voyageai.com/docs/pricing) ·
[Cohere Embed v4](https://docs.cohere.com/changelog/embed-multimodal-v4) ·
[jina-embeddings-v4](https://arxiv.org/abs/2506.18902) · [jina-embeddings-v5-text](https://arxiv.org/abs/2602.15547) ·
[EmbeddingGemma](https://arxiv.org/pdf/2509.20354) ·
[Arctic-Embed 2.0](https://arxiv.org/abs/2412.04506) ·
[BGE-M3](https://huggingface.co/BAAI/bge-m3) · [BGE-Reasoner](https://huggingface.co/BAAI/bge-reasoner-embed-qwen3-8b-0923) ·
[MMTEB (ICLR 2025)](https://arxiv.org/abs/2502.13595) ·
[SPLADE-v3](https://arxiv.org/abs/2403.06789) · [SPLADE at billion scale](https://arxiv.org/abs/2511.22263) ·
[OpenSearch neural sparse v3](https://opensearch.org/blog/advancing-search-with-opensearch-v3-neural-sparse-models-and-a-multilingual-retrieval-model/) ·
[ELSER v2](https://www.elastic.co/search-labs/blog/introducing-elser-v2-part-1) · [BM25S](https://github.com/xhluca/bm25s) ·
[ColBERTv2](https://arxiv.org/pdf/2112.01488) · [PLAID](https://arxiv.org/pdf/2205.09707) · [MUVERA](https://arxiv.org/abs/2405.19504) ·
[Qdrant on MUVERA](https://qdrant.tech/articles/muvera-embeddings/) · [Vespa, scaling ColPali](https://blog.vespa.ai/scaling-colpali-to-billions/) ·
[ColPali (ICLR 2025)](https://arxiv.org/abs/2407.01449) · [ViDoRe V3](https://arxiv.org/abs/2601.08620) ·
[Nemotron ColEmbed V2](https://arxiv.org/abs/2602.03992) · [ColModernVBERT](https://arxiv.org/abs/2510.01149) ·
[Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) ·
[VisRAG](https://arxiv.org/abs/2410.10594) · [M3DocRAG](https://arxiv.org/abs/2411.04952) ·
[Lost in OCR Translation?](https://arxiv.org/abs/2505.05666) ·
[Document-as-Image Representations Fall Short](https://arxiv.org/abs/2604.18508) ·
[Visual RAG Toolkit](https://arxiv.org/abs/2602.12510) · [ColChunk](https://arxiv.org/abs/2604.10167) · [PIXELRAG](https://arxiv.org/abs/2606.28344)

**Fusion, reranking, routing**
[Bruch, Gai & Ingber, Fusion Functions](https://arxiv.org/abs/2210.11934) ·
[Balancing the Blend](https://arxiv.org/abs/2508.01405) ·
[Vespa, Embedding Tradeoffs Quantified](https://blog.vespa.ai/embedding-tradeoffs-quantified/) ·
[Akarsu et al., From BM25 to Corrective RAG](https://arxiv.org/abs/2604.01733) ·
[T²-RAGBench](https://arxiv.org/abs/2506.12071) ·
[Drowning in Documents](https://arxiv.org/abs/2411.11767) ·
[mxbai-rerank-v2](https://www.mixedbread.com/blog/mxbai-rerank-v2) ·
[jina-reranker-v3](https://arxiv.org/abs/2509.25085) · [jina-reranker-v3.5](https://arxiv.org/abs/2607.18152) ·
[Voyage rerank-2.5](https://blog.voyageai.com/2025/08/11/rerank-2-5/) · [Cohere Rerank 4](https://cohere.com/blog/rerank-4) ·
[FIRST (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.491/) ·
[How Good are LLM-based Rerankers?](https://arxiv.org/abs/2508.16757) ·
[Rank1](https://arxiv.org/abs/2502.18418) · [Rank-K](https://arxiv.org/abs/2505.14432) · [ReasonRank](https://arxiv.org/abs/2508.07050) · [DIVER](https://arxiv.org/abs/2508.07995) ·
[Are LLM-Based Retrievers Worth Their Cost? (SIGIR 2026)](https://arxiv.org/abs/2604.03676) ·
[Compute Allocation for Reasoning-Intensive Retrieval Agents](https://arxiv.org/abs/2603.14635) ·
[BRIGHT](https://arxiv.org/abs/2407.12883) · [Lighting the Way for BRIGHT](https://arxiv.org/abs/2509.02558) · [ReasonIR](https://arxiv.org/abs/2504.20595) · [BRIGHT-Pro](https://arxiv.org/abs/2605.04018) ·
[Better Together: Complementary Query Rewriting](https://arxiv.org/abs/2609.05637) ·
[When Should Queries Be Decomposed?](https://arxiv.org/abs/2606.08577) ·
[Adaptive-RAG](https://arxiv.org/abs/2403.14403) · [RAGRouter-Bench](https://arxiv.org/abs/2604.03455)

**Long context**
[Self-Route](https://arxiv.org/abs/2407.16833) · [Databricks long-context RAG](https://arxiv.org/abs/2411.03538) ·
[NoLiMa](https://arxiv.org/abs/2502.05167) · [Chroma, Context Rot](https://www.trychroma.com/research/context-rot) ·
[LongBench v2](https://arxiv.org/abs/2412.15204) · [Long Context vs. RAG: Evaluation and Revisits](https://arxiv.org/abs/2501.01880)

**Evaluation**
[TREC 2025 RAG overview](https://arxiv.org/abs/2603.09891) · [UMBRELA](https://arxiv.org/abs/2406.06519) ·
[Clarke & Dietz](https://arxiv.org/abs/2412.17156) · [Human-in-the-Loop Nugget Annotation](https://arxiv.org/abs/2606.29033) ·
[FreshStack](https://arxiv.org/abs/2504.13128) · [RAGBench](https://arxiv.org/abs/2407.11005) ·
[Sakai, Topic set size design](https://dl.acm.org/doi/10.1007/s10791-015-9273-z)

**Store internals and lifecycle**
[HAKARI-Bench](https://arxiv.org/abs/2606.22778) · [Disk-resident graph ANN evaluation](https://arxiv.org/abs/2603.01779) ·
[RACORN-1](https://arxiv.org/html/2607.00768) · [Qdrant, filtered search](https://qdrant.tech/articles/filtered-vector-search-acorn/) ·
[Qdrant optimizer](https://qdrant.tech/documentation/concepts/optimizer/) · [Milvus compaction](https://milvus.io/blog/2022-2-21-compact.md) ·
[Ghost Vectors](https://arxiv.org/abs/2606.18497) · [Drift-Adapter](https://arxiv.org/abs/2509.23471)

**Claim audit**
[Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory](https://arxiv.org/abs/2603.02473)

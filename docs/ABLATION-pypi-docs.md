# Ablation report: PyPI documentation corpus

> **Superseded by [`ABLATION.md`](ABLATION.md).** This is the first, eight-arm
> run on the PyPI-docs corpus (18 Sep 2026), kept for the record. Its numbers
> predate eight fixes to the measurement apparatus -- among them an empty
> structured result scored as a success and a router comparing string fields
> against floats -- so they are not comparable with the current report. The
> `check_index_surface` false positive it found is fixed and still in force.

**Run date:** 2026-09-18. **Config:** `configs/pypi-docs.yaml`. **Command:**

```bash
python scripts/fetch_corpus.py
python scripts/run_ablation.py configs/pypi-docs.yaml
```

This is the first ablation of the frame on a real corpus, and it is an
*offline* run: no model download, no API key. That constraint decides what the
numbers can and cannot mean, so it comes first.

## What was run

| | |
|---|---|
| Corpus | 686 documents from 50 PyPI source distributions (README, `docs/**`, changelogs); 550 reStructuredText, 104 Markdown, 32 plain text; 10.2 MB parsed |
| Units | 10,283 structural units (400-token ceiling, 48-token floor, no overlap); 7,918 for the fixed-window arm |
| Golden set | 333 queries, machine-bootstrapped, **unverified**: 300 factual, 16 structured, 13 temporal, 4 numeric; one gold span each |
| Dense index | `hash_embedding`: hashed word and character-4-gram features, 384 dimensions, exact cosine. **Not a semantic model.** |
| Lexical index | `bm25_memory`, k1 = 1.2, b = 0.75 |
| Structured index | `sqlite` over regex-extracted `version`, `version_major`, `release_date` and the scanner-supplied `package` |
| Contextualiser | `extractive_context`: two lead sentences of the parent document plus six distinctive terms, ≤ 320 chars. The offline stand-in for an LLM summary. |
| Reranker | `lexical_overlap`: query-term coverage, proximity, prior rank. The offline stand-in for a cross-encoder. |
| Router | `rules` |
| Judge | `containment` (the gold snippet appears in the top-5 concatenation) |
| Wall time | 815 s for the first full run including the 76 s base build; 211 s to re-run the ladder after one arm changed, because the ledger skipped the 686 unchanged documents in every other arm |

## The delta table

Deltas against arm 1. `<` is worse than the baseline, `>` better. `fail%` is
the share of queries with nothing relevant in the top 20.

```
arm                     P@5             R@20            nDCG@10         fail%           correct         p50ms   p95ms
-----------------------------------------------------------------------------------------------------------------------
1-lexical-only          0.145           0.917           0.573           0.075           0.730           17      21
2-dense-only            0.054 (-0.091)< 0.473 (-0.443)< 0.209 (-0.364)< 0.474 (+0.399)< 0.345 (-0.384)<  8      10
3-hybrid-rrf            0.105 (-0.041)< 0.847 (-0.070)< 0.411 (-0.161)< 0.138 (+0.063)< 0.607 (-0.123)< 27      36
4-hybrid-sectionpath    0.102 (-0.043)< 0.820 (-0.097)< 0.393 (-0.180)< 0.162 (+0.087)< 0.598 (-0.132)< 41      51
5-hybrid-context        0.087 (-0.058)< 0.790 (-0.127)< 0.329 (-0.243)< 0.189 (+0.114)< 0.535 (-0.195)< 30      37
6-hybrid-context-rerank 0.103 (-0.043)< 0.853 (-0.063)< 0.406 (-0.167)< 0.132 (+0.057)< 0.616 (-0.114)< 35      42
7-no-router             0.093 (-0.052)< 0.790 (-0.127)< 0.378 (-0.194)< 0.210 (+0.135)< 0.625 (-0.105)< 43      66
8-fixed-window          0.105 (-0.041)< 0.777 (-0.140)< 0.405 (-0.167)< 0.201 (+0.126)< 0.643 (-0.087)< 33      42

[INVESTIGATE] contextualisation cuts retrieval failures by ~1/3: retrieval_failure_rate 0.138 -> 0.189 (+37%, expected -33% ±50%)
[PASS]        reranking roughly halves them again:               retrieval_failure_rate 0.189 -> 0.132 (-30%, expected -50% ±50%)
```

Retrieval failure rate by query type (the slice invariant 5 lives in):

| arm | factual (n=300) | structured (16) | temporal (13) | numeric (4) | route accuracy |
|---|---|---|---|---|---|
| 1–6, 8 | 0.083 – 0.527 | 0.000 | 0.000 | 0.000 | 1.00 |
| 7-no-router | 0.157 | **1.000** | 0.231 | **1.000** | 0.00 |

## What the numbers say

Read in order. Each finding names the arms it rests on.

1. **BM25 alone wins this ladder, and that is a fact about the golden set as
   much as about BM25.** The bootstrapper builds each query from a unit's
   subject and three distinctive terms, then *drops any query BM25 cannot find
   in its top 50*. Gold is therefore BM25-findable by construction, and the
   queries are keyword strings ("flask after arguments assign"), not questions.
   Arm 1's 7.5% failure rate is the ceiling this set can show, not a statement
   about lexical retrieval on real traffic. A verified, human-written or
   log-mined set (`GoldOrigin.HUMAN` / `PRODUCTION_LOG`) is the prerequisite for
   any absolute claim.

2. **A weak leg drags fusion down (arms 1, 2, 3).** The hashed dense index fails
   on 47% of queries; fusing it with BM25 by RRF takes the failure rate from
   7.5% to 13.8% and nDCG@10 from 0.573 to 0.411. This is the "weakest link"
   result of *Balancing the Blend* (2025) reproduced on a small corpus, and it
   is why `docs/STATE_OF_THE_ART.md` §6 says a leg has to earn its place. It is
   **not** evidence against hybrid retrieval: the dense leg here is not a
   semantic model. It is evidence that the frame should carry a leg-quality
   gate, and that "hybrid beats either half" (invariant 4) presupposes two
   competent halves.

3. **Document-level context hurt unit-level questions (arms 3, 4, 5).** Adding
   the heading trail cost 2.4 points of failure rate; adding two lead sentences
   and six document-distinctive terms cost 5.1. Every query in this set targets
   one unit inside a multi-unit document, so a prefix shared by all of a
   document's units makes them *less* distinguishable from each other. That is
   exactly the in-document degradation Zhou et al. report (*Beyond
   Chunk-Then-Embed*, Feb 2026), and it is why the first sanity check reads
   INVESTIGATE. The diagnosis text ("check the wiring") is the right first
   step and was taken: a term that appears only in a unit's *context* and in
   two surfaces corpus-wide (`2026-03-19`) returns that unit at rank 1 from
   the lexical index and rank 20 from the dense one, so both index the
   contextualised string. The wiring is fine; the effect on this corpus and
   this query mix is negative. The published −⅓ is for cross-document
   questions over a real LLM summary, and this run had neither.

4. **Reranking recovered most of what context cost (arms 5, 6).** A
   lexical-overlap reranker with no model took the failure rate from 18.9% to
   13.2% (−30%) and nDCG@10 from 0.329 to 0.406. The second sanity check
   passes. The expected halving is calibrated for a cross-encoder; a smaller
   effect from a term-overlap heuristic is the honest outcome, and it lands
   inside the stated tolerance.

5. **Routing is worth exactly what invariant 5 predicts, and it is invisible
   in the aggregate (arms 6, 7).** Disabling the router moves the headline
   failure rate from 13.2% to 21.0%. The per-type slice shows where: every
   structured and numeric question fails (1.000), temporal questions fail
   23%, and factual questions barely move (0.147 → 0.157). Thirty-three
   structured questions out of 333 cost 8 points overall and are catastrophic
   for their slice. This is the case for `RunReport.by_query_type`.

6. **Structure beats fixed windows with overlap (arms 6, 8).** Same context,
   same reranker; swapping the structural segmenter for 400-token windows with
   64-token overlap raises the failure rate from 13.2% to 20.1% with 23% fewer
   units. Consistent with the 2026 chunking comparisons (Bennani & Moslonka;
   Shaukat et al.; Śmigielski et al.): structure-aligned boundaries win, and
   overlap does not compensate.

7. **Incrementality works (build log).** The re-run after changing one arm
   rebuilt that arm (24 s) and skipped 686 unchanged documents in every other
   arm: 211 s for eight arms against 815 s cold. Invariant 2 measured.

## What the harness caught

The first run of arm 4 reported route accuracy 0.0 and every structured
question failing, which made no sense for an arm that only changes the context
prefix. The cause was in the arm's override, not in the code: lists replace on
overlay, so `ingestion.enrich.enrichers: [section_prefix]` silently removed
the field extractor and with it the structured path. The arm was measuring two
changes while claiming one, which is the failure mode invariant 6 exists to
prevent. Fixed by restating `regex_fields` in the override, with a comment
explaining why. The per-type slice is what made the bug visible; the aggregate
alone would have read as "section prefixes are bad".

A second small defect was in the reporting: a passing sanity check within
tolerance but below the expected magnitude carried the diagnosis "larger than
expected". `_diagnose` now returns an empty diagnosis for anything inside the
tolerance band.

The third was in the checker the diagnosis points at. `check_index_surface`
picked the *first* context-only term it found and searched for it with
`top_k=50`. On this corpus that term was "Changelog", present in 2,181 of the
10,283 retrieval surfaces, and the check reported both indexes as indexing raw
text when neither was. The probe is now the context-only term with the lowest
document frequency over the surface, searched at a depth of at least that
frequency. A checker that fails on a correct system on the first real corpus is
worse than no checker, because the diagnosis it feeds is trusted.

## What this run can and cannot establish

**Established.** The wiring is correct (contextualisation reaches both indexes;
reranking reorders the right candidate set; the router sends structured
questions to the structured index and the harness sees it when it does not).
The direction of every arm-to-arm delta is explicable by the 2025–26
literature. Span-anchored gold survived a segmenter change (arm 8) without
edits. The ledger makes re-runs cheap.

**Not established.** Any absolute number. The dense index is not a semantic
model, the reranker is not a cross-encoder, the contextualiser is not an LLM,
and the golden set is machine-generated and lexically biased. Validating the
published magnitudes needs `configs/full.yaml`, credentials for the
contextualiser, a cross-encoder download, and a golden set with at least a
human-verified subset. Nothing else in the frame has to change for that run.

**To add before the next run.** Paired-bootstrap confidence intervals on each
delta; at n = 333 with one gold span per query, differences of two or three
points are within noise and the table should say so. Rerank `input_top_k` at
100 as well as 50, per the 2026 depth findings. A `hybrid-context-rrf-k20`
arm (already in `configs/reference.yaml`) to see whether the fusion constant
moves the weak-leg result.

## Files

- `var/pypi/eval/ablation.json`, `var/pypi/eval/ablation.txt`: the run's own
  output (not committed; `var/` is a build product).
- `data/pypi-docs-golden.jsonl`: the golden set, committed so the run is
  reproducible. `data/pypi-docs/_corpus.json` pins every package version.

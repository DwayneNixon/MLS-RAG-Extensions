# Machine Learning Systems — Joint Coursework (Submission Draft)

**INFR11269 · University of Edinburgh · Autumn 2026/27**

This private repository assembles coursework contributions by **Dwayne Nixon and Nitay Carmi**. It combines CUDA automatic speech recognition (ASR), retrieval-augmented generation (RAG), technical systems investigations, experimental evidence, and the official report resources.

**Status: integration draft, NOT a final course submission.** The report template still requires the actual joint report, group details, AI-use declaration, and compiled PDF. Read [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) before hand-in.

## Navigation

| Area | Files |
|---|---|
| Four CUDA kernel tasks | [hw1-asr/csrc/](hw1-asr/csrc/) |
| Alternative CUDA source and new GEMM investigation | [hw1-asr/extensions/](hw1-asr/extensions/) |
| ASR test / profiling evidence | [hw1-asr/results/](hw1-asr/results/) |
| Audio-to-answer RAG baseline | [rag/](rag/) |
| Retrieval-depth, hybrid reranking, MiniLM and evaluation code | [rag/extensions/](rag/extensions/) |
| Saved query, judge and systems results | [rag/results/](rag/results/) |
| Report template, format checker and 326-number audit | [report/](report/) |
| Exact merge and source-version decisions | [INTEGRATION_PROVENANCE.md](INTEGRATION_PROVENANCE.md) |

## CUDA investigations

The default five `hw1-asr/csrc/` task files are preserved **unaltered from Nitay's source branch**; its saved final 33-case test log and GPU identification are in `hw1-asr/results/`. Dwayne's independently validated source versions are retained separately in `hw1-asr/extensions/dwayne_variant/csrc/` and on his unchanged source branch.

The additional register-tiled, shape-aware GEMM **v2** lives at `hw1-asr/extensions/gemm_v2/task3_gemm.cu`; its instructions and evidence are documented in [hw1-asr/extensions/README.md](hw1-asr/extensions/README.md). It is **not automatically the default `csrc/task3_gemm.cu`**. The saved experiment reports 1,563.6 ms/transcription for v2 vs 2,689.8 ms for its v1 comparison (~1.72× faster than v1), still slower than its separate PyTorch benchmark. A later block-size sweep includes other experimental settings. Do not relabel a variant as the submitted kernel without a verified full build/test.

## RAG investigations

- Dense retrieval baseline, context-depth k=2/5/10/20, reference judged-answer evaluation, and additional Practice depth replication.
- Practice-tuned Hybrid BM25/weighted RRF (dense weight 0.95) and pretrained MiniLM cross-encoder (top 100 -> top 5).
- Complete Development retrieval (125 queries) and common judged comparisons (123 queries). Neither reranker established an overall judged-answer quality gain relative to k5 in the recorded paired analysis.
- Dwayne's matched-node reranking-only test: 369 calls per method on the same RTX 2080 Ti node; Hybrid CPU mean 22.09 ms versus MiniLM GPU mean 280.52 ms (12.70× observed latency ratio). This excludes generator and upstream stages.
- Nitay's Practice depth replication: for k2 minus k5, n=347 mean -0.00768, one-sided 95% lower bound -0.03393, so non-inferiority at the prespecified 0.03 margin was **not established on Practice**; pooled n=470 lower bound -0.02683 is supportive, not an independent confirmatory outcome.
- See `rag/results/` and `report/report_numbers.txt`: the saved audit checked **326 values, 0 mismatches, 0 missing result files** against the draft report numbers.

These are **distinct experiments and timing protocols**; do not mix Dwayne's ASR benchmark times with Nitay's as if they were a controlled paired run.

## Reproduction and data handling

The original framework is [ed-aisys/edin-mls-26-autumn](https://github.com/ed-aisys/edin-mls-26-autumn). Start from [hw1-asr/README.md](hw1-asr/README.md) and [rag/README.md](rag/README.md). The course's `rag-assets.zip` is managed via Git LFS; install it as the course describes. The corpus, model weights and ELM credentials are **not** self-contained in this source tree. Configure each person's ELM key **locally** in ignored `rag/.env`; never commit an actual key. A pretrained external MiniLM checkpoint was used in research and needs proper acknowledgement and final course-policy review.

The report must use the supplied LaTeX/Overleaf template without changing its style. The six-page main-body limit excludes references and optional appendices; essential arguments must fit in the main body. Complete the template's explicit AI declaration and group-information section.

**Branch sources:** Dwayne `45215acf37713659536adcda491556024d5058ef`; Nitay `5ee6cf8f42e980af128ec16f12baf75613e46d93`. No original branch has been rewritten. See [INTEGRATION_PROVENANCE.md](INTEGRATION_PROVENANCE.md).

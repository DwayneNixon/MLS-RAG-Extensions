# Machine Learning Systems — Supplementary Code

**INFR11269 · University of Edinburgh · Autumn 2026/27**

This repository combines the supplied coursework framework with completed CUDA automatic speech recognition (ASR) tasks, retrieval-augmented generation (RAG) investigations and their recorded evidence. **All course-provided CUDA tutorials, framework files and report templates are retained.**


## Start here

- [Experiment index](EXPERIMENT_INDEX.md) — an annotated guide to sources, investigations and supporting results.
- [Assigned CUDA implementations and verification](hw1-asr/FINAL_CUDA.md) — current task files and the 17 November check.
- [RAG extensions guide](rag/extensions/README.md) — retrieval depth, reranking, answer-length and latency experiments.
- [Report resources](report/README.md) — the official LaTeX template and checker. The completed report will be added later.

## Repository map

| Area | Files |
|---|---|
| Original CUDA tutorials (retained) | [cuda-tutorial/](cuda-tutorial/) |
| Four assigned CUDA kernel tasks | [hw1-asr/csrc/](hw1-asr/csrc/) |
| Alternative CUDA source and new GEMM investigation | [hw1-asr/extensions/](hw1-asr/extensions/) |
| ASR test / profiling evidence | [hw1-asr/results/](hw1-asr/results/) |
| Audio-to-answer RAG baseline | [rag/](rag/) |
| Retrieval-depth, hybrid reranking, MiniLM and evaluation code | [rag/extensions/](rag/extensions/) |
| Saved query, judge and systems results | [rag/results/](rag/results/) |
| Report template, checker and earlier-draft number audit | [report/](report/) |
| Experiment-by-experiment index | [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) |

## CUDA investigations

The five standard `hw1-asr/csrc/` task files are the **completed assigned kernels** (Tasks 1-4), including the completed tile-32 shared-memory GEMM in `csrc/task3_gemm.cu`. This is the code the coursework tests (`make test`) and the report's "original GEMM" rows describe. An earlier draft of this branch had replaced `csrc/task3_gemm.cu` with the register-tiled GEMM (v2, BK=32); that replacement has been **reverted** (17 November) so the task file is the completed task again.

The register-tiled GEMMs are **extensions**, built separately and never replacing the task file: `hw1-asr/extensions/gemm_v2/` (4x4 per thread, BK=16, tile height by shape) and `hw1-asr/extensions/gemm_v3/` (register block and tile depth chosen by shape). Their tests, racecheck and same-session timings are in `hw1-asr/extensions/results/` (block sweep: `gemm_blocks_3716786/`; v3, M sweep and longer clips: `followup_kernels_*` and `followup_clips_*`). Final validation of the assigned task files was completed on **17 November 2026** in `hw1-asr/extensions/results/verify_submission_3727463/`: **33/33 tests PASS**, exact ASR transcript **PASS**, and **0 racecheck hazards** in the three small GEMM cases tested. The assigned-kernel profile recorded **2716.1 ± 9.1 ms/transcription** on RTX 2080 Ti. The register-tiled GEMMs remain separate extensions.

**Distinguish the CUDA measurement snapshots.** [The 17 November verification](hw1-asr/extensions/results/verify_submission_3727463/summary.txt) is for the **currently assigned tile-32 GEMM** (33/33 tests, transcription PASS, **2716.1 ± 9.1 ms**). The earlier [`submission_profile.log`](hw1-asr/results/submission_profile.log) (**1468.6 ± 2.1 ms**) and [paired PyTorch reference](hw1-asr/results/submission_pytorch_baseline.log) (**1081.1 ± 0.8 ms**) are from an **older register-tiled BK32 draft in the default task path**, not from the current assigned GEMM. Separate controlled extension comparisons are retained in [`hw1-asr/extensions/results/`](hw1-asr/extensions/results/).

## RAG investigations

- Dense retrieval baseline, context-depth k=2/5/10/20, reference judged-answer evaluation, and additional Practice depth replication.
- Practice-tuned Hybrid BM25/weighted RRF (dense weight 0.95) and pretrained MiniLM cross-encoder (top 100 -> top 5).
- Complete Development retrieval (125 queries) and common judged comparisons (123 queries). Neither reranker established an overall judged-answer quality gain relative to k5 in the recorded paired analysis.
- The matched-node reranking-only test: 369 calls per method on the same RTX 2080 Ti node; Hybrid CPU mean 22.09 ms versus MiniLM GPU mean 280.52 ms (12.70× observed latency ratio). This excludes generator and upstream stages.
- The Practice depth replication: for k2 minus k5, n=347 mean -0.00768, one-sided 95% lower bound -0.03393, so non-inferiority at the prespecified 0.03 margin was **not established on Practice**; pooled n=470 lower bound -0.02683 is supportive, not an independent confirmatory outcome.
- The answer-length study and smoke test are retained in [`phase5/answer_length/`](rag/results/phase5/answer_length/) and [`phase5/answer_length_smoke/`](rag/results/phase5/answer_length_smoke/), respectively.
- See `rag/results/` and `report/report_numbers.txt`: the saved file records an **audit of an earlier report draft** (not yet re-run against a final report).

These are **distinct experiments and timing protocols**; do not mix the two sets of ASR benchmark times as if they were a controlled paired run.

The bracketed note in `rag/extensions/ANALYSIS_PLAN_practice_depth.md` is a leftover drafting prompt; the file is unchanged since it was committed.

## Reproduction and data handling

The original framework is [ed-aisys/edin-mls-26-autumn](https://github.com/ed-aisys/edin-mls-26-autumn). Start from [hw1-asr/README.md](hw1-asr/README.md) and [rag/README.md](rag/README.md). The course's `rag-assets.zip` is managed via Git LFS; install it as the course describes. The corpus, model weights and ELM credentials are **not** self-contained in this source tree. Configure each person's ELM key **locally** in ignored `rag/.env`; never commit an actual key. A pretrained external MiniLM checkpoint was used in research and needs proper acknowledgement and final course-policy review.

The report must use the supplied LaTeX/Overleaf template without changing its style. The six-page main-body limit excludes references and optional appendices; essential arguments must fit in the main body. Complete the template's explicit AI declaration and group-information section.


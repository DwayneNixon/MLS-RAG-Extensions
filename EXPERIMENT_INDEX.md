# Experiment index — supplementary code

This index points to the **assigned source files**, **experimental alternatives** and **saved evidence**. The supplied [CUDA tutorials](cuda-tutorial/), [ASR framework](hw1-asr/), [RAG framework](rag/) and [report template](report/) are retained unchanged by this organisation pass.

## CUDA ASR

| Investigation | Source and results | How to interpret |
|---|---|---|
| Assigned four CUDA tasks | [`hw1-asr/csrc/`](hw1-asr/csrc/), [`FINAL_CUDA.md`](hw1-asr/FINAL_CUDA.md) | Includes assigned tile-32 GEMM |
| Final verification of current task files | [17 November summary](hw1-asr/extensions/results/verify_submission_3727463/summary.txt), [full logs](hw1-asr/extensions/results/verify_submission_3727463/) | **33/33 PASS**, transcription PASS, **0 hazards in three small racecheck cases**, **2716.1 ± 9.1 ms** |
| Register-tiled GEMM extensions | [v2](hw1-asr/extensions/gemm_v2/task3_gemm.cu), [v3](hw1-asr/extensions/gemm_v3/task3_gemm.cu), [documentation](hw1-asr/extensions/README.md) | Extensions **do not** replace the assigned GEMM |
| Historical same-session GEMM block sweep | [`gemm_blocks_3716786/summary.txt`](hw1-asr/extensions/results/gemm_blocks_3716786/summary.txt) | Compare variants only within the recorded session |
| Follow-up GEMM tuning and GPU profiler counters | [`followup_kernels_3727089/summary.txt`](hw1-asr/extensions/results/followup_kernels_3727089/summary.txt) | Tile variants, bank conflicts, racecheck, occupancy |
| Longer-clip comparison | [`followup_clips_3727464/summary.txt`](hw1-asr/extensions/results/followup_clips_3727464/summary.txt) | PyTorch, assigned, v2 and v3 on matched clips |
| Prior task experiments | [`hw1-asr/results/`](hw1-asr/results/) | `submission_profile.log` is from an **earlier BK32 draft**, not current assigned-kernel validation |

## RAG

| Investigation | Code | Results |
|---|---|---|
| Retrieval depth and answer quality | [Retrieval-k](rag/extensions/evaluate_retrieval_k.py), [phase3 analysis](rag/extensions/analyse_phase3.py) | [`retrieval_k_dev.jsonl`](rag/results/retrieval_k_dev.jsonl), [`phase3/`](rag/results/phase3/) |
| Hybrid BM25/RRF reranking | [Hybrid reranker](rag/extensions/hybrid_reranker.py), [alpha sweep](rag/extensions/sweep_hybrid_alpha.py) | [Practice sweep](rag/results/hybrid_alpha_sweep_practice.csv), [Development ranking](rag/results/hybrid_dev_alpha095.jsonl) |
| MiniLM neural reranking | [Cross-encoder](rag/extensions/cross_encoder_reranker.py), [evaluation](rag/extensions/evaluate_minilm.py) | [Development ranking](rag/results/minilm_dev_full125.jsonl), [judgements](rag/results/minilm_dev_judge_full125/) |
| Reranking effectiveness and timing | [Unified analysis](rag/extensions/analyse_reranking_unified.py), [matched benchmark](rag/extensions/benchmark_reranking_matched.py) | [123-query comparison](rag/results/reranking_unified_dev123.json), [timings](rag/results/reranking_matched_damnii_3705581.json) |
| Oracle evidence/context order | [Phase4 runner](rag/extensions/run_phase4.sh) | [`phase4/`](rag/results/phase4/) |
| Practice k2 versus k5 | [Prespecified plan](rag/extensions/ANALYSIS_PLAN_practice_depth.md), [analysis](rag/extensions/analyse_practice_depth.py) | [Practice outputs](rag/results/practice_depth/); non-inferiority **not established** |
| Answer length | [Plan](rag/extensions/ANALYSIS_PLAN_answer_length.md), [analysis](rag/extensions/analyse_answer_length.py) | [Full run](rag/results/phase5/answer_length/), [smoke test](rag/results/phase5/answer_length_smoke/) |

For supporting utilities and interpretation boundaries see [the RAG guide](rag/extensions/README.md).

## Report, assets and comparison rules

The final LaTeX report will be added after it is completed. The [official template and checker](report/) remain available; [`report_numbers.txt`](report/report_numbers.txt) audits an **earlier report draft** and should be rerun. The checked-in [`rag-assets.zip`](rag-assets.zip) is a Git LFS pointer, not the extracted course data.

**Never mix GPU measurements from different jobs as if they were a controlled comparison, or confuse reranking-only latency with end-to-end RAG latency.**

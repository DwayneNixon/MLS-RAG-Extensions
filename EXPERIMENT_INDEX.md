# Supplementary code — experiment index

Use this index to locate **implementations**, **evaluation scripts**, and **saved measurements**. The supplied [CUDA tutorials](cuda-tutorial/), [ASR framework](hw1-asr/), [RAG framework](rag/), and [report resources](report/) remain part of the repository.

## CUDA / ASR

| Investigation | Source and evidence | Scope |
|---|---|---|
| Assigned CUDA tasks | [`hw1-asr/csrc/`](hw1-asr/csrc/), [task descriptions](hw1-asr/FINAL_CUDA.md) | RoPE, reductions, tile-32 GEMM, attention |
| Assigned-kernel correctness and ASR | [Verification summary](hw1-asr/extensions/results/verify_submission_3727463/summary.txt), [complete logs](hw1-asr/extensions/results/verify_submission_3727463/) | **33/33 PASS**, transcription PASS, **2716.1 ± 9.1 ms**, no hazards in small-case GEMM racecheck |
| Register-tiled GEMM implementations | [v2](hw1-asr/extensions/gemm_v2/task3_gemm.cu), [v3](hw1-asr/extensions/gemm_v3/task3_gemm.cu), [guide](hw1-asr/extensions/README.md) | Alternative builds separate from the assigned task source |
| GEMM block sweep | [`gemm_blocks_3716786/summary.txt`](hw1-asr/extensions/results/gemm_blocks_3716786/summary.txt) | Same-job comparisons of tile/register configurations |
| GEMM shape tuning and GPU profiling | [`followup_kernels_3727089/summary.txt`](hw1-asr/extensions/results/followup_kernels_3727089/summary.txt) | Occupancy, bank conflicts, racecheck, tile and BK configurations |
| ASR across longer clips | [`followup_clips_3727464/summary.txt`](hw1-asr/extensions/results/followup_clips_3727464/summary.txt) | PyTorch, assigned CUDA, v2 and v3 |
| Additional CUDA profiles | [`hw1-asr/results/`](hw1-asr/results/) | Includes **BK32 register-tiled** `submission_profile.log`; not the assigned tile-32 profile |

## RAG

| Investigation | Code | Evidence |
|---|---|---|
| Dense retrieval depth and answer quality | [Retrieval-k](rag/extensions/evaluate_retrieval_k.py), [analysis](rag/extensions/analyse_phase3.py) | [Development retrieval](rag/results/retrieval_k_dev.jsonl), [phase3 results](rag/results/phase3/) |
| Hybrid BM25/dense fusion | [Hybrid reranker](rag/extensions/hybrid_reranker.py), [alpha sweep](rag/extensions/sweep_hybrid_alpha.py) | [Practice sweep](rag/results/hybrid_alpha_sweep_practice.csv), [Development ranking](rag/results/hybrid_dev_alpha095.jsonl) |
| MiniLM reranking | [Cross-encoder](rag/extensions/cross_encoder_reranker.py), [evaluation](rag/extensions/evaluate_minilm.py) | [Development ranking](rag/results/minilm_dev_full125.jsonl), [judge outputs](rag/results/minilm_dev_judge_full125/) |
| Reranker quality and systems cost | [Unified analysis](rag/extensions/analyse_reranking_unified.py), [benchmark](rag/extensions/benchmark_reranking_matched.py) | [Common-query analysis](rag/results/reranking_unified_dev123.json), [matched timings](rag/results/reranking_matched_damnii_3705581.json) |
| Oracle evidence and context order | [Phase 4 runner](rag/extensions/run_phase4.sh) | [Phase 4 results](rag/results/phase4/) |
| Practice depth replication | [Prespecified plan](rag/extensions/ANALYSIS_PLAN_practice_depth.md), [analysis](rag/extensions/analyse_practice_depth.py) | [Practice results](rag/results/practice_depth/) |
| Answer-length experiment | [Plan](rag/extensions/ANALYSIS_PLAN_answer_length.md), [analysis](rag/extensions/analyse_answer_length.py) | [Full experiment](rag/results/phase5/answer_length/), [smoke test](rag/results/phase5/answer_length_smoke/) |

The [RAG extensions guide](rag/extensions/README.md) describes the additional generation/judging scripts, prerequisites, experimental boundaries, and limitations.

## Reproduction and reporting

Environment setup is documented in the [ASR guide](hw1-asr/README.md) and [RAG guide](rag/README.md). The tracked [`rag-assets.zip`](rag-assets.zip) is an LFS pointer; course assets and external pretrained checkpoints must be installed separately.

The [report directory](report/) contains LaTeX resources and the number audit. Regenerate [`report_numbers.txt`](report/report_numbers.txt) against the final submitted report.

**Interpretation:** distinguish the assigned GEMM from the separately compiled extensions; compare latency only under comparable timing protocols and workloads.

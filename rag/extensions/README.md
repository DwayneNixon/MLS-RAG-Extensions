# RAG extensions — investigation guide

This directory contains investigations built on the [supplied RAG framework](../README.md). Course scripts, tests and the original pipeline remain intact. Experiment code is here; saved measurements and judge outputs are under [`../results/`](../results/). See also the [coursework experiment index](../../EXPERIMENT_INDEX.md).

## Experiment map

| Investigation | Main code and protocol | Saved evidence |
|---|---|---|
| Dense retrieval depth and answer quality | [`evaluate_retrieval_k.py`](evaluate_retrieval_k.py), [`run_phase3.sh`](run_phase3.sh), [`analyse_phase3.py`](analyse_phase3.py) | [Development retrieval](../results/retrieval_k_dev.jsonl), [Practice retrieval](../results/retrieval_k_practice.jsonl), [phase3](../results/phase3/) |
| Hybrid BM25 + dense fusion | [`hybrid_reranker.py`](hybrid_reranker.py), [`sweep_hybrid_alpha.py`](sweep_hybrid_alpha.py), [`evaluate_hybrid.py`](evaluate_hybrid.py) | [Practice alpha sweep](../results/hybrid_alpha_sweep_practice.csv), [Development ranking](../results/hybrid_dev_alpha095.jsonl), [judge results](../results/hybrid_dev_judge_full125/) |
| MiniLM neural reranking | [`cross_encoder_reranker.py`](cross_encoder_reranker.py), [`evaluate_minilm.py`](evaluate_minilm.py) | [Development ranking](../results/minilm_dev_full125.jsonl), [judge results](../results/minilm_dev_judge_full125/) |
| Unified reranker comparison and matched latency | [`analyse_reranking_unified.py`](analyse_reranking_unified.py), [`benchmark_reranking_matched.py`](benchmark_reranking_matched.py) | [Common-query analysis](../results/reranking_unified_dev123.json), [matched time](../results/reranking_matched_damnii_3705581.json) |
| Oracle evidence and context order | [`run_phase4.sh`](run_phase4.sh) | [phase4](../results/phase4/) |
| Practice retrieval-depth replication | [`ANALYSIS_PLAN_practice_depth.md`](ANALYSIS_PLAN_practice_depth.md), [`run_practice_depth.sh`](run_practice_depth.sh), [`analyse_practice_depth.py`](analyse_practice_depth.py) | [Practice depth results](../results/practice_depth/) |
| Answer length versus cost/quality | [`ANALYSIS_PLAN_answer_length.md`](ANALYSIS_PLAN_answer_length.md), [`generate_length_arms.py`](generate_length_arms.py), [`run_answer_length.sh`](run_answer_length.sh), [`analyse_answer_length.py`](analyse_answer_length.py) | [Full answer-length run](../results/phase5/answer_length/), [smoke test](../results/phase5/answer_length_smoke/) |
| Full-pipeline latency instrumentation | [`e2e_latency.py`](e2e_latency.py), [`run_e2e_latency.sbatch`](run_e2e_latency.sbatch) | Scripts are present; no separate committed full-pipeline latency result directory in this snapshot |

## Shared utilities

[Cached generation](generate_answers_cached.py), [resumable generation](generate_answers_resumable.py), [reranked generation](generate_answers_reranked.py), [resumable judging](judge_resumable.py), [selective Hybrid generation](generate_hybrid_dev_selective.py) and [MiniLM generation](generate_minilm_dev_resumable.py) support the experiments above. [`run_phase3_judge.sh`](run_phase3_judge.sh) orchestrates judging; check the runner scripts before execution.

## Interpreting results

- The Development split contains 125 queries; the common paired judged-answer analysis includes **123**. A retrieval recall improvement does not automatically improve judged answer quality.
- The matched reranking benchmark is **stage-only**, excluding upstream dense retrieval, ELM generation and judging. Its saved means are **22.09 ms Hybrid CPU** versus **280.52 ms MiniLM GPU**, with 369 timed calls per method in that protocol.
- The Practice replication of **k=2 versus k=5** used a prespecified non-inferiority margin of **0.03**. Non-inferiority was **not established** on the saved Practice result.
- Full answer-length results and smoke-test outputs are separate, not interchangeable experiments. Cross-job GPU timings require caution.

Both analysis plans are retained **without edits**. The bracketed note in the Practice plan is a historical drafting prompt.

For setup, dependencies and ELM-key safety see [`rag/README.md`](../README.md). The external pretrained MiniLM checkpoint and the full course assets are not included in the source checkout. [`report_numbers.txt`](../../report/report_numbers.txt) audits an earlier draft; rerun [`report_numbers.py`](../../report/report_numbers.py) when the actual report is final.

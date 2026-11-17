# Machine Learning Systems — Supplementary Code

**INFR11269 · University of Edinburgh · 2026/27**

This repository contains the supplied coursework framework, completed CUDA automatic speech recognition (ASR) kernels, retrieval-augmented generation (RAG) investigations, and supporting experimental evidence. Course-provided CUDA tutorials, tests, and report resources are retained.

## Start here

| Area | What to inspect |
|---|---|
| [Experiment index](EXPERIMENT_INDEX.md) | Overview of experiments and links to implementation and evidence |
| [CUDA task source](hw1-asr/csrc/) | Assigned RoPE, reductions, tiled GEMM, and attention |
| [CUDA extensions](hw1-asr/extensions/README.md) | Register-tiled GEMM variants, benchmarks, profiling, and checks |
| [CUDA verification](hw1-asr/FINAL_CUDA.md) | Test results, performance measurements, and reproduction commands |
| [RAG extensions](rag/extensions/README.md) | Retrieval depth, reranking, answer length, and evaluation |
| [Report resources](report/README.md) | LaTeX source/template, checking tools, and number-audit script |

## Repository structure

| Path | Contents |
|---|---|
| [`cuda-tutorial/`](cuda-tutorial/) | Supplied CUDA tutorials |
| [`hw1-asr/csrc/`](hw1-asr/csrc/) | Assigned CUDA task implementations |
| [`hw1-asr/extensions/`](hw1-asr/extensions/) | Alternative GEMM implementations, benchmarking code, and recorded experiments |
| [`hw1-asr/results/`](hw1-asr/results/) | Kernel-level testing and ASR profiling output |
| [`rag/src/`](rag/src/) | RAG pipeline implementation |
| [`rag/extensions/`](rag/extensions/) | Investigation-specific scripts and evaluation code |
| [`rag/results/`](rag/results/) | Retrieval, answer generation, judging, and latency results |
| [`report/`](report/) | Official report resources and numerical audit |

## CUDA / ASR investigation

The assigned task implementation in [`hw1-asr/csrc/`](hw1-asr/csrc/) uses a **tile-32 shared-memory GEMM** alongside custom RoPE, reductions, and attention kernels. The [register-tiled GEMM v2](hw1-asr/extensions/gemm_v2/task3_gemm.cu) and [v3](hw1-asr/extensions/gemm_v3/task3_gemm.cu) are **separate experimental implementations**.

For reproducibility, the saved results identify the exact kernel configuration being tested:

| Configuration | Evidence | Result |
|---|---|---|
| Assigned CUDA task files, including tile-32 GEMM | [Task verification](hw1-asr/extensions/results/verify_submission_3727463/summary.txt) | **33/33 tests PASS**, transcription PASS, **2716.1 ± 9.1 ms/transcription**; small-case GEMM racecheck: **0 hazards** |
| Register-tiled BK32 CUDA configuration | [ASR profile](hw1-asr/results/submission_profile.log) | Transcription PASS, **1468.6 ± 2.1 ms/transcription** |
| PyTorch reference measured with the BK32 configuration | [PyTorch profile](hw1-asr/results/submission_pytorch_baseline.log) | Transcription PASS, **1081.1 ± 0.8 ms/transcription** |

All figures are from recorded RTX 2080 Ti experiments. The register-tiled BK32 profile is **not a measurement of the assigned tile-32 task files**. Only compare times measured under compatible workloads and benchmark conditions. Controlled v2/v3 comparisons and longer-clip experiments are indexed in [`hw1-asr/extensions/README.md`](hw1-asr/extensions/README.md).

## RAG investigation

The RAG work evaluates how retrieval depth, evidence ranking, and generation settings affect answer quality and system cost.

- **Retrieval depth:** dense top-k retrieval and answer comparisons for k=2, 5, 10, and 20.
- **Reranking:** Hybrid BM25/dense fusion and a pretrained MiniLM cross-encoder, both reranking a dense candidate pool.
- **Quality analysis:** paired judged answers across **123 common Development queries**; neither reranker established an overall answer-quality improvement in this comparison.
- **Systems cost:** matched reranking-only benchmarks (369 calls per method) measured **22.09 ms** mean for Hybrid CPU and **280.52 ms** for MiniLM GPU. These are stage-only measurements, not full-pipeline latency.
- **Practice replication:** the prespecified k=2 versus k=5 non-inferiority comparison did **not** establish non-inferiority at the 0.03 margin.
- **Answer length:** saved generation-length and quality/cost investigations, including separate full and smoke-test records.

See the [RAG experiment guide](rag/extensions/README.md) and [experiment index](EXPERIMENT_INDEX.md) for the relevant code and result files.

## Reproduction and data dependencies

Start with [`hw1-asr/README.md`](hw1-asr/README.md) and [`rag/README.md`](rag/README.md). The source tree includes the Git LFS pointer for `rag-assets.zip`, not the extracted corpus, recordings, embeddings, or model weights. Install the supplied course assets using the documented setup process.

RAG generation requires locally configured ELM access, and neural reranking uses an external pretrained MiniLM checkpoint. Do not commit API keys or local credential files.

The [report directory](report/) contains the official LaTeX resources, format checker, and [`report_numbers.py`](report/report_numbers.py). The saved [`report_numbers.txt`](report/report_numbers.txt) is an existing numerical audit and should be regenerated against the final report to establish consistency.

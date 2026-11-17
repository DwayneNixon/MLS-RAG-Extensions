# Integration provenance and decisions — 14 November 2026

> **Amendment, 17 November 2026:** the GEMM promotion described below has been **reverted**. `hw1-asr/csrc/task3_gemm.cu` is again the completed assigned tile-32 GEMM (restored byte for byte from the archived copy, which has been removed because the task file now holds it). The register-tiled GEMMs are extensions only (`hw1-asr/extensions/gemm_v2/`, `gemm_v3/`). See `hw1-asr/FINAL_CUDA.md`.

## Exact source commits

- Dwayne: [MLS-Dwayne](https://github.com/DwayneNixon/MLS-RAG-Extensions/tree/MLS-Dwayne), `45215acf37713659536adcda491556024d5058ef` (9 November 2026).
- Nitay: [MLS-Nitay](https://github.com/DwayneNixon/MLS-RAG-Extensions/tree/MLS-Nitay), `5ee6cf8f42e980af128ec16f12baf75613e46d93` (13 November 2026).
- Existing empty main baseline: `5201ccf56c85bd6010c4045faf7ac33b87fbbc63`.

Source inventories from Git trees: Dwayne **129 tracked blobs**, Nitay **310**. **119 identical file blobs**, **10 same-path but different blobs**, **181 files present only in Nitay**; **none present only in Dwayne**. GitHub reports **no common ancestor** between the two contributor branches. They must not be blindly merged with an ordinary Git merge.

## Assembly method

- The complete Nitay Git tree is the starting snapshot because it already contains 181 additional result/code assets, the report-number audit, the Practice depth replication and GEMM v2 experiments.
- Initially the duplicate paths retained Nitay's version. **Subsequent CUDA finalisation** promoted the one-variable BK=32 register-tiled v2 into `hw1-asr/csrc/task3_gemm.cu`; the other four default task files still retain Nitay's content. The original Nitay v1 GEMM is preserved separately. This selection reflects the saved 13 November comparative benchmark, not a fresh post-integration GPU run.
- Copy the five Dwayne-version CUDA task file blobs, without modification, under `hw1-asr/extensions/dwayne_variant/csrc/`; they remain independently accessible on Dwayne's original branch.
- Keep Dwayne's previous research-focused README under `docs/dwayne_original_readme_2026-11-09.md`, and his earlier differing RAG scripts under `rag/extensions/archive_dwayne/` to retain byte-for-byte source provenance.
- Keep the tested BK=16 GEMM v2 extension for historical reproducibility and the selected BK=32 GEMM source at the standard task path. Full same-node integration verification remains an open gate; the performance-selection decision itself is complete.
- Replace only the top-level README with the joint, evidence-qualified overview, and add this provenance file and the checklist.
- Do **not** alter the original two branches, force-push, silently invent benchmark output, or treat a draft PR as a course submission.

## Source conflict index

The ten initial same-path differences (before the GEMM v2 promotion) were: `README.md`; all five `hw1-asr/csrc/` task files; `rag/extensions/analyse_phase3.py`; `rag/extensions/generate_answers_cached.py`; `rag/extensions/run_matched_damnii.sbatch`; and `rag/extensions/run_phase3.sh`.

For actual tests and output provenance, consult `hw1-asr/results/`, `hw1-asr/extensions/results/`, `rag/results/`, Dwayne's 9 November findings and Nitay's 13 November `report/report_numbers.txt`. The current new assembly has **not** been rebuilt or GPU-tested as a unified checkout.

## CUDA promotion decision (14 November 2026)

The file `hw1-asr/csrc/task3_gemm.cu` now contains the entire original `hw1-asr/extensions/gemm_v2/task3_gemm.cu` source, with **only `constexpr int BK = 16;` changed to `constexpr int BK = 32;`**. This reproduces the exact tested one-line variant described by `hw1-asr/extensions/results/gemm_blocks_3716786/test_tm4_tn4_bn64_bk32.txt` and its corresponding full-ASR run. The former `hw1-asr/csrc/task3_gemm.cu` blob is preserved **byte for byte** at `hw1-asr/extensions/nitay_v1_task3_gemm.cu`. The test logs were produced from an isolated `/tmp` build using the same surrounding Nitay CUDA files; the new promoted checkout has not independently been rebuilt or benchmarked. Never cite the old full-suite `final_test.txt` as proof that BK=32 was included in that 33/33 run.

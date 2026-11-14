# Integration provenance and decisions — 14 November 2026

## Exact source commits

- Dwayne: [MLS-Dwayne](https://github.com/DwayneNixon/MLS-RAG-Extensions/tree/MLS-Dwayne), `45215acf37713659536adcda491556024d5058ef` (9 November 2026).
- Nitay: [MLS-Nitay](https://github.com/DwayneNixon/MLS-RAG-Extensions/tree/MLS-Nitay), `5ee6cf8f42e980af128ec16f12baf75613e46d93` (13 November 2026).
- Existing empty main baseline: `5201ccf56c85bd6010c4045faf7ac33b87fbbc63`.

Source inventories from Git trees: Dwayne **129 tracked blobs**, Nitay **310**. **119 identical file blobs**, **10 same-path but different blobs**, **181 files present only in Nitay**; **none present only in Dwayne**. GitHub reports **no common ancestor** between the two contributor branches. They must not be blindly merged with an ordinary Git merge.

## Assembly method

- The complete Nitay Git tree is the starting snapshot because it already contains 181 additional result/code assets, the report-number audit, the Practice depth replication and GEMM v2 experiments.
- Files with the same path retain Nitay's version in the working tree, including the five default `hw1-asr/csrc` task files. This is an **integration decision**, not a conclusion that the alternative implementation is worse.
- Copy the five Dwayne-version CUDA task file blobs, without modification, under `hw1-asr/extensions/dwayne_variant/csrc/`; they remain independently accessible on Dwayne's original branch.
- Keep Dwayne's previous research-focused README under `docs/dwayne_original_readme_2026-11-09.md`, and his earlier differing RAG scripts under `rag/extensions/archive_dwayne/` to retain byte-for-byte source provenance.
- Keep the separate newer GEMM-v2 alternative as an extension and explicitly flag the default-kernel selection as unresolved until final verification.
- Replace only the top-level README with the joint, evidence-qualified overview, and add this provenance file and the checklist.
- Do **not** alter the original two branches, force-push, silently invent benchmark output, or treat a draft PR as a course submission.

## Source conflict index

The ten same-path differences are: `README.md`; all five `hw1-asr/csrc/` task files; `rag/extensions/analyse_phase3.py`; `rag/extensions/generate_answers_cached.py`; `rag/extensions/run_matched_damnii.sbatch`; and `rag/extensions/run_phase3.sh`.

For actual tests and output provenance, consult `hw1-asr/results/`, `hw1-asr/extensions/results/`, `rag/results/`, Dwayne's 9 November findings and Nitay's 13 November `report/report_numbers.txt`. The current new assembly has **not** been rebuilt or GPU-tested as a unified checkout.

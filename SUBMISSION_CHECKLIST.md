# Joint MLS Coursework — Submission Checklist

This is a working inventory, not evidence of a completed hand-in. Official material: `hw1-asr/README.md`, `report/README.md`, the course RAG guide, and the 2026 Week 4 coursework slides. The exact Moodle/Learn upload mechanics, deadline, and permitted external-checkpoint policy require confirmation from the live course brief.

## Code

- [x] Five task-source files are present at `hw1-asr/csrc/task1_rope.cu`, `task2_reduce.cuh`, `task2_reduce.cu`, `task3_gemm.cu` and `task4_attention.cu` (Nitay sources for Tasks 1/2/4; GEMM promoted from tested v2 BK=32).
- [x] Dwayne's different five-file implementation is preserved at `hw1-asr/extensions/dwayne_variant/csrc/`.
- [x] Register-tiled/shape-aware GEMM v2 and its separately recorded tests are at `hw1-asr/extensions/gemm_v2/` and `hw1-asr/extensions/results/`.
- [x] Select **register-tiled GEMM v2 with BK=32** as the final CUDA source candidate, promoted into `hw1-asr/csrc/task3_gemm.cu`. Original Nitay v1 and Dwayne's variant are archived. **Fresh integrated validation still outstanding**.
- [ ] Verify that the chosen five-file snapshot is what the final submitted report describes; do not modify course-provided Python, provided kernels or tests in the hand-in.

## Required ASR evidence and discussion

- [x] Saved `make test` transcript: `hw1-asr/results/final_test.txt` (four task suites ALL PASS on RTX 2080 Ti).
- [x] Saved profiling output: `hw1-asr/results/final_profile.txt`, `final_baseline_profile.txt` and related measurements.
- [x] Saved GPU identification: `hw1-asr/results/final_gpu.txt` (RTX 2080 Ti).
- [x] Task-specific Try This measurements and negative results are preserved in `hw1-asr/results/`, `hw1-asr/extensions/results/`, and Dwayne's `FINDINGS.md` project notes.
- [ ] **Required final gate**: on a single RTX 2080 Ti GPU node, clean-build the new standard `csrc/` version, confirm **33/33 tests PASS**, run `./run.sh --profile` with Status PASS, and save `nvidia-smi -L` from that same node. The existing full 33-case log is from v1, while BK=32's 10-case and transcription logs were generated in the isolated test variant. See `hw1-asr/FINAL_CUDA.md`.
- [ ] Incorporate answers to all guide report questions and at least one measured Try This per task into the joint report or allowed supporting material.

## RAG experiment / reproducibility evidence

- [x] Baseline corpus/ASR bridge, encoder, retriever, generator, scripts, requirements and tests.
- [x] Saved Practice (350) and Development (125) evaluation outputs and relevant per-query results.
- [x] k-depth, oracle/context-order, Hybrid BM25/RRF and MiniLM investigations, answer/judge outputs and analysis code.
- [x] Matched same-node reranking benchmark script and `reranking_matched_damnii_3705581.json` with 369 timings/method.
- [x] Practice depth replication, prespecified analysis plan and `analysis_summary.json`.
- [x] `report/report_numbers.py` and `report/report_numbers.txt` (326 values checked, 0 differences and 0 absent input files in the saved audit).
- [ ] Explain and cite the external pretrained MiniLM checkpoint if its use is permitted under course policy.
- [ ] State the non-confirmatory result of the Practice non-inferiority test, judge stochasticity, reused answers, qrels limits, and comparable vs incomparable timing boundaries accurately.

## Final written deliverables

- [x] Official LaTeX source template, Overleaf ZIP and format checker available in `report/`.
- [ ] Add the actual **joint** `report.tex`, references and figures (existing `report/latex_template/report.tex` is still the teaching **placeholder**).
- [ ] Use `\author{Anonymous coursework submission}`, preserve review formatting and do not enable `\iclrfinalcopy`.
- [ ] Keep main-body content at **six pages maximum**; references/optional appendix excluded, but key evidence must be in body.
- [ ] Fill in the template's **AI declaration** accurately, including AI-assisted code, text, analysis or graphics.
- [ ] Fill in the **group information** page with the actual group details (do not leave the template's sample names or Group ID XXX).
- [ ] Compile PDF and run `python report/format_checker/check_report_format.py /path/to/report.pdf`; inspect warnings and figures.
- [ ] Verify source/figure/table numbers against saved output and rerun `python report/report_numbers.py` once final prose is fixed.

## Security and hand-in checks

- [x] This assembled snapshot does not contain a committed `rag/.env` by its tree inventory.
- [ ] Run final sensitive-data scan of files and *history*, especially experiment logs, generated answer files, and transcript caches, before any public release. Keep this repository private.
- [ ] Check the exact Learn/course submission method and due date; do not assume GitHub is itself the submission.
- [ ] Get both contributors' final review before merging this draft to `main` or uploading the actual hand-in.

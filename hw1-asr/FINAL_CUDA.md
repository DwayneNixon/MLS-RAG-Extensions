# CUDA task implementations and verification

## Assigned task files

The files in `hw1-asr/csrc/` implement the required CUDA operations:

| File | Implementation |
|---|---|
| `csrc/task1_rope.cu` | RoPE, one thread per element pair, 256 threads per block |
| `csrc/task2_reduce.cuh`, `csrc/task2_reduce.cu` | Warp-shuffle and shared-memory reductions for RMSNorm, LayerNorm, and stable softmax |
| `csrc/task3_gemm.cu` | **Tile-32 shared-memory GEMM**, one output per thread, padded weight tile, fused bias/GELU |
| `csrc/task4_attention.cu` | Streaming online-softmax attention with causal masking and grouped-query attention |

The assigned `task3_gemm.cu` is **not** the register-tiled extension. The distinction is relevant when reproducing the reported benchmark results.

## Experimental GEMM extensions

| Path | Implementation |
|---|---|
| [`extensions/gemm_v2/`](extensions/gemm_v2/) | Register-tiled GEMM: 4×4 outputs per thread, BK=16, shape-dependent tile height |
| [`extensions/gemm_v3/`](extensions/gemm_v3/) | Parameterised register tiling: 8×4 for larger M, 4×4 for smaller M, configurable BK |
| [`extensions/variant_b/`](extensions/variant_b/) | Alternative set of task-source implementations retained for comparison |

Each register-tiled extension is built separately with the supplied Makefile and its own output directory; it does not replace the assigned task source. For example, from `hw1-asr/` on an RTX 2080 Ti session:

```bash
make SRC=extensions/gemm_v3 BUILD=/tmp/$USER/build_v3 lib
ASR_LIB=/tmp/$USER/build_v3/libasr_kernels.so ./run.sh --profile
```

## Verification and measurements

| Configuration | Location | Result |
|---|---|---|
| Assigned task sources (tile-32 GEMM) | [`extensions/results/verify_submission_3727463/`](extensions/results/verify_submission_3727463/) | **33/33 tests PASS**, transcription PASS; **2716.1 ± 9.1 ms/transcription**, 8.56 GiB peak GPU memory; **0 hazards** in the three small GEMM racecheck cases |
| Register-tiled BK32 configuration in the ASR framework | [`results/submission_profile.log`](results/submission_profile.log) | Transcription PASS; **1468.6 ± 2.1 ms/transcription** |
| PyTorch-only reference, paired with the BK32 test | [`results/submission_pytorch_baseline.log`](results/submission_pytorch_baseline.log) | Transcription PASS; **1081.1 ± 0.8 ms/transcription** |

The assigned-kernel result and register-tiled BK32 result are **different kernel configurations and runs**. Treat each set of timings according to its own recorded benchmark protocol; do not infer a direct paired improvement across sessions.

Additional controlled comparisons are in [`extensions/results/gemm_blocks_3716786/`](extensions/results/gemm_blocks_3716786/), [`extensions/results/followup_kernels_3727089/`](extensions/results/followup_kernels_3727089/) and [`extensions/results/followup_clips_3727464/`](extensions/results/followup_clips_3727464/).

## Reproducing assigned task validation

From `hw1-asr/` on the DICE login node:

```bash
sbatch extensions/run_verify_submission.sbatch
```

The job runs on one GPU and records GPU identity, a clean `make test` of the four task groups (4 + 11 + 10 + 8 = 33), full-ASR transcription and the small-case GEMM racecheck. Consult the resulting `extensions/results/verify_submission_JOBID/` output and the supplied Makefile/README for environment setup.

# CUDA hand-in: what is submitted and how it is validated

**Updated:** 17 November 2026. **Branch:** `MLS-Submission-Draft`.

## Files to hand in

The five task source files in `hw1-asr/csrc/` are the completed assigned kernels:

| File | Implementation |
|---|---|
| `csrc/task1_rope.cu` | RoPE, one thread per element pair, 256 threads per block |
| `csrc/task2_reduce.cuh`, `csrc/task2_reduce.cu` | warp-shuffle + shared-memory reductions (RMSNorm, LayerNorm, stable softmax), 256 threads |
| `csrc/task3_gemm.cu` | **completed tile-32 shared-memory GEMM** (one output per thread, padded weight tile, fused bias/GELU) |
| `csrc/task4_attention.cu` | streaming online-softmax attention with causal masking and GQA, 16 warps |

An earlier draft of this branch (14 November) replaced `csrc/task3_gemm.cu` with the register-tiled GEMM v2 at BK=32. That was **reverted on 17 November**: the task file is the completed task, byte for byte the version behind `results/final_test.txt` (33/33) and `results/final_profile.txt` (2,689.8 ms).

## Extensions (built separately, never replacing a task file)

| Folder | What it is |
|---|---|
| `extensions/gemm_v2/` | register-tiled GEMM, 4x4 outputs per thread, BK=16, 64- or 32-row tiles by shape |
| `extensions/gemm_v3/` | the same kernel with the register block and tile depth as template parameters: 8x4 for M >= 128, 4x4 below; BK set at compile time (`-DV3_BK_ENC`, `-DV3_BK_DEC`) |

Every other file in these folders is a symbolic link to `csrc/`, so an extension build differs from the task build only in the GEMM. Build and run one with:

```bash
make SRC=extensions/gemm_v3 BUILD=/tmp/$USER/build_v3 lib
ASR_LIB=/tmp/$USER/build_v3/libasr_kernels.so ./run.sh --profile
```

Evidence: `extensions/results/gemm_blocks_3716786/` (block-size sweep, one job), `extensions/results/followup_kernels_*/` (M sweep, profiler counters, v2 against v3: tests, racecheck on both dispatch paths, timing) and `extensions/results/followup_clips_*/` (longer clips). Compare timings only within one job.

## Final validation of the task files

From `hw1-asr` on the head node:

```bash
sbatch extensions/run_verify_submission.sbatch
cat extensions/results/verify_submission_JOBID/summary.txt
```

One job, one GPU: GPU identity, a clean `make test` (expect 4 + 11 + 10 + 8 = 33 PASS), `./run.sh --profile` (expect Status PASS), and GEMM racecheck on the test binary.

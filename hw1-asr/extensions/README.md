> **Integration update (14 November 2026):** This page records the historical *separately built* GEMM v2 experiments. The final CUDA hand-in **candidate** now places the tested v2 register-tiled algorithm with `BK=32` at the normal `hw1-asr/csrc/task3_gemm.cu` path. The original v1 remains at `hw1-asr/extensions/nitay_v1_task3_gemm.cu`, while the original v2 BK=16 remains in this extension directory. The 13 November BK=32 sweep passed 10/10 GEMM cases and the full-model transcription; integrated 33/33 tests and same-node profile are still to be run. See [`../FINAL_CUDA.md`](../FINAL_CUDA.md).

# HW1 extension: register-tiled GEMM (v2)

The submitted kernels in `hw1-asr/csrc/` are unchanged. This folder holds a second
version of the GEMM kernel only, built and run separately, so v1 and v2 can be
compared on identical code for every other operation.

## Why

Profiling the v1 kernels showed the tiled GEMM takes 89% of GPU time and reaches
only 16-31% of cuBLAS throughput. v1 computes one output per thread, so every
multiply-add needs one value from shared memory. Measuring calls by shape
(`results/v1_profile_by_M.txt`) showed only 13 of 2,757 GEMM calls have a single
row (the LM head, 1.8% of GEMM time); the decoder recomputes the full sequence
each step (M of about 59-71) and the encoder uses M = 1500. So the fix targets
data reuse for these shapes, not a single-row path.

## What changed (`gemm_v2/task3_gemm.cu`)

| Change | Reason |
|---|---|
| Each thread computes a 4 x 4 block of outputs in registers (outer products) | 16 multiply-adds per 8 values loaded, instead of 1 per value |
| 256 threads per block, BN = 64, BK = 16 | Smaller blocks than v1's 1,024 threads |
| Shared-memory tiles padded by one column | Removes bank conflicts in the cooperative loads |
| Tile height chosen by shape: BM = 64 if M >= 128, else 32 (template parameter) | Large M favours reuse (64); the decoder's M of about 64 needs more blocks to fill 68 SMs (32) |

Bias and GELU are applied exactly as in v1. The v1 kernel is still in the file but
no longer called. Every other source file in `gemm_v2/` is a symbolic link to
`csrc/`, so only the GEMM differs.

## How to build and run

From `hw1-asr/`, in the HW1 environment, on a GPU node:

    V2="SRC=extensions/gemm_v2 BUILD=/tmp/$USER/build_v2"
    make $V2 test3                                   # kernel tests against cuBLAS
    make $V2                                         # build the library
    ASR_LIB=/tmp/$USER/build_v2/libasr_kernels.so ./run.sh --profile   # whole model

Build into `/tmp`: linking on the shared filesystem can hang.

## Results (RTX 2080 Ti)

Kernel tests, GFLOP/s (higher is better):

| Shape (M x N x K) | v1 | 64x64 | 32x64 | 32x64 padded | 64x64 padded | Final (by shape) | cuBLAS |
|---|---|---|---|---|---|---|---|
| Decoder q_proj 64x2048x2048 | 1,440 | 1,686 | 1,743 | 1,999 | 1,793 | 1,999 | 4,685 |
| Decoder gate_proj 64x6144x2048 | 1,496 | 2,517 | 2,148 | 3,299 | 3,085 | 3,269 | 6,696 |
| Decoder down_proj 64x2048x6144 | 1,804 | 1,891 | 1,807 | 1,994 | 1,802 | 1,992 | 6,996 |
| Encoder q_proj 1500x1280x1280 | 2,137 | 3,844 | 3,209 | 4,208 | 5,307 | 5,304 | 12,973 |
| Encoder fc1 + GELU 1500x5120x1280 | 2,111 | 3,971 | 3,131 | 3,721 | 5,254 | 5,297 | 12,957 |
| Encoder conv2 1500x1280x3840 | 2,140 | 3,821 | 3,188 | 5,024 | 5,143 | 5,190 | 13,463 |
| LM head 1x59264x2048 | 59 | 61 | 74 | 122 | 72 | 122 | 290 |

Whole model, one 3.5 s utterance, mean of three runs after one warm-up:

| Version | ms per transcription | GEMM GPU time |
|---|---|---|
| PyTorch baseline | 768.9 | 572 ms |
| v1 (submitted kernels) | 2,688.0 | 2,419 ms |
| v2, 64x64 padded everywhere | 1,644.7 | 1,351 ms |
| v2, tile chosen by shape (final) | 1,563.6 | 1,271 ms |

All kernel tests pass, the transcription matches the reference, and racecheck
reports 0 hazards (first six launches of each tile version, including the
odd-sized boundary cases).

## Results files (`results/`)

| File | What it shows |
|---|---|
| `v1_profile_by_M.txt` | v1 profile with single-row GEMM calls labelled separately |
| `v2_test_before.txt` | the v2 build pipeline reproduces v1 exactly |
| `v2_regtile_64x64.txt`, `v2_regtile_32x64.txt` | register tiling, unpadded |
| `v2_regtile_32x64_pad.txt`, `v2_regtile_64x64_pad.txt` | register tiling, padded |
| `v2_racecheck.txt`, `v2_dispatch_racecheck.txt` | racecheck, 0 hazards |
| `v2_e2e_64x64_pad.txt` | whole model, 64x64 padded |
| `v2_dispatch_test.txt`, `v2_e2e_dispatch.txt` | final version: kernel tests and whole model |

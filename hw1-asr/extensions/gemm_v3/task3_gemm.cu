// Task 3: the linear layer as a tiled GEMM. Read docs/task3-gemm.md first.
//
//     y[M, N] = act( x[M, K] * w[N, K]^T + bias[N] )
//
// This is the tiled GEMM from cuda-tutorial/5-tiled-gemm, with three changes:
//   1. w is stored the way PyTorch stores Linear weights: one row per OUTPUT
//      feature, [N, K]. So column n of the result uses row n of w.
//   2. bias is added, and GELU can be applied, before the result is stored
//      (the "epilogue"), so no extra kernel has to read y again.
//   3. every Linear layer of the ASR model and both convolutions run through it.
//
// Block (TILE x TILE threads) computes a TILE x TILE piece of y. Thread (ty, tx)
// computes y[row][col] with row = blockIdx.y*TILE + ty, col = blockIdx.x*TILE + tx.
// The block walks along K one tile at a time, staging a tile of x and a tile of w
// in shared memory so every loaded value is used TILE times.
#include "asr_kernels.h"
#include "common.cuh"

constexpr int TILE = 32;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)

__global__ void linear_tiled_kernel(const float* x, const float* w, const float* bias,
                                    float* y, int M, int N, int K, int act) {
  __shared__ float xs[TILE][TILE];      // xs[m][k]: rows of x for this block
  __shared__ float ws[TILE][TILE + 1];  // ws[n][k]: rows of w for this block (+1: see Try this)

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;  // the output row this thread computes
  const int col = blockIdx.x * TILE + tx;  // the output column this thread computes

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // 3.1: load this thread's element of the x tile (0 outside x)
    if (row < M && k0 + tx < K) {
      xs[ty][tx] = x[row * K + (k0 + tx)];
    } else {
      xs[ty][tx] = 0.0f;
    }

    // 3.2: load this thread's element of the w tile; consecutive tx read consecutive k
    int w_row = blockIdx.x * TILE + ty;
    int w_col = k0 + tx;
    if (w_row < N && w_col < K) {
      ws[ty][tx] = w[w_row * K + w_col];
    } else {
      ws[ty][tx] = 0.0f;
    }

    __syncthreads();  // wait until every thread has loaded both tiles

    // 3.3: this K step's contribution
    for (int k = 0; k < TILE; k += 1) {
      acc += xs[ty][k] * ws[tx][k];
    }

    __syncthreads();  // wait until everyone has used the tiles before they are overwritten
  }

  // 3.4: epilogue
  if (row < M && col < N) {
    if (bias != nullptr) {
      acc += bias[col];
    }
    if (act == ACT_GELU) {
      acc = gelu(acc);
    }
    y[row * N + col] = acc;
  }
}


// ---- v3: register-tiled GEMM with the register block and tile depth chosen by shape ----
// Same kernel body as v2 (extensions/gemm_v2), but TM, TN and BK are template parameters
// alongside BM, so each shape gets its own block. The block-size sweep
// (results/gemm_blocks_3716786) found 8x4 best on the encoder and 4x4 best on the decoder.
// y[M,N] = act(x[M,K] * w[N,K]^T + bias)
constexpr int BN = 64;
#ifndef V3_BK_ENC
#define V3_BK_ENC 16   // tile depth for M >= 128 (encoder)
#endif
#ifndef V3_BK_DEC
#define V3_BK_DEC 32   // tile depth for M < 128 (decoder)
#endif

template <int BM, int TM, int TN, int BK>
__global__ void linear_regtile_kernel(const float* x, const float* w, const float* bias,
                                      float* y, int M, int N, int K, int act) {
  constexpr int NT = (BM / TM) * (BN / TN);   // threads per block
  __shared__ float xs[BK][BM + 1];
  __shared__ float ws[BK][BN + 1];

  const int tx = threadIdx.x % (BN / TN);
  const int ty = threadIdx.x / (BN / TN);
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;

  float acc[TM][TN] = {0.0f};
  float a[TM];
  float b[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = threadIdx.x; i < BM * BK; i += NT) {      // load x tile
      int r = i / BK, c = i % BK;
      int gr = row0 + r, gk = k0 + c;
      xs[c][r] = (gr < M && gk < K) ? x[gr * K + gk] : 0.0f;
    }
    for (int i = threadIdx.x; i < BN * BK; i += NT) {      // load w tile
      int r = i / BK, c = i % BK;
      int gn = col0 + r, gk = k0 + c;
      ws[c][r] = (gn < N && gk < K) ? w[gn * K + gk] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {                      // outer products in registers
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = xs[kk][ty * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = ws[kk][tx * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)                             // epilogue, same as v1 and v2
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int r = row0 + ty * TM + i, c = col0 + tx * TN + j;
      if (r < M && c < N) {
        float v = acc[i][j];
        if (bias != nullptr) v += bias[c];
        if (act == ACT_GELU) v = gelu(v);
        y[r * N + c] = v;
      }
    }
}

template <int BM, int TM, int TN, int BK>
static void launch_regtile(const float* x, const float* w, const float* bias, float* y,
                           int M, int N, int K, int act, cudaStream_t stream) {
  dim3 block((BM / TM) * (BN / TN));
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  linear_regtile_kernel<BM, TM, TN, BK><<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
}

ASR_API int launch_linear(const float* x, const float* w, const float* bias, float* y,
                          int M, int N, int K, int act, cudaStream_t stream) {
  // tall matrices (encoder, M = 1500): 64-row tiles, 8x4 outputs per thread (128 threads);
  // short ones (decoder, M ~ 59-71, and M = 1): 32-row tiles, 4x4 per thread (128 threads)
  if (M >= 128) {
    launch_regtile<64, 8, 4, V3_BK_ENC>(x, w, bias, y, M, N, K, act, stream);
  } else {
    launch_regtile<32, 4, 4, V3_BK_DEC>(x, w, bias, y, M, N, K, act, stream);
  }
  RETURN_LAUNCH_ERROR();
}

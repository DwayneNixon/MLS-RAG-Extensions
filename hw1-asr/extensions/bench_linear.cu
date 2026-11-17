// Throughput of launch_linear over a list of M at fixed N and K (default 2048 x 2048,
// the decoder q_proj shape). Linked against one kernel variant's objects, like the
// supplied tests. Used by run_followup_gpu.sbatch for the M sweep.
//
//   bench_linear [N] [K] M1 M2 ...      e.g. bench_linear 2048 2048 16 32 64 96 128 256 512
//
// Prints one line per M: M, N, K, median ms over 20 launches (after 3 warm-ups), GFLOP/s.
// BENCH_BM / BENCH_TM / BENCH_TN (set at compile time) only label the output and give
// the grid and threads per block the variant launches.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "asr_kernels.h"

#ifndef BENCH_BM
#define BENCH_BM 0
#endif
#ifndef BENCH_TM
#define BENCH_TM 0
#endif
#ifndef BENCH_TN
#define BENCH_TN 0
#endif
#ifndef BENCH_BN
#define BENCH_BN 64
#endif

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  std::exit(1); } } while (0)

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: %s N K M1 [M2 ...]\n", argv[0]); return 2; }
  const int N = std::atoi(argv[1]), K = std::atoi(argv[2]);
  int maxM = 0;
  for (int i = 3; i < argc; ++i) maxM = std::max(maxM, std::atoi(argv[i]));

  std::vector<float> hx((size_t)maxM * K), hw((size_t)N * K);
  unsigned s = 12345;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 65536.0f - 0.5f; };
  for (auto& v : hx) v = rnd();
  for (auto& v : hw) v = rnd();
  float *x, *w, *y;
  CHECK(cudaMalloc(&x, hx.size() * sizeof(float)));
  CHECK(cudaMalloc(&w, hw.size() * sizeof(float)));
  CHECK(cudaMalloc(&y, (size_t)maxM * N * sizeof(float)));
  CHECK(cudaMemcpy(x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(w, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));

  std::printf("variant BM=%d TM=%d TN=%d BN=%d\n", BENCH_BM, BENCH_TM, BENCH_TN, BENCH_BN);
  std::printf("%6s %6s %6s %10s %10s %8s %8s\n", "M", "N", "K", "ms", "GFLOP/s", "blocks", "threads");
  for (int i = 3; i < argc; ++i) {
    const int M = std::atoi(argv[i]);
    for (int r = 0; r < 3; ++r) CHECK((cudaError_t)launch_linear(x, w, nullptr, y, M, N, K, 0, 0));
    CHECK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int r = 0; r < 20; ++r) {
      CHECK(cudaEventRecord(a));
      CHECK((cudaError_t)launch_linear(x, w, nullptr, y, M, N, K, 0, 0));
      CHECK(cudaEventRecord(b));
      CHECK(cudaEventSynchronize(b));
      float ms;
      CHECK(cudaEventElapsedTime(&ms, a, b));
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    const float ms = t[t.size() / 2];
    const int blocks = BENCH_BM ? ((N + BENCH_BN - 1) / BENCH_BN) * ((M + BENCH_BM - 1) / BENCH_BM) : 0;
    const int threads = (BENCH_BM && BENCH_TM) ? (BENCH_BM / BENCH_TM) * (BENCH_BN / BENCH_TN) : 0;
    std::printf("%6d %6d %6d %10.4f %10.1f %8d %8d\n", M, N, K, ms, 2.0 * M * N * K / (ms * 1e6), blocks, threads);
  }
  return 0;
}

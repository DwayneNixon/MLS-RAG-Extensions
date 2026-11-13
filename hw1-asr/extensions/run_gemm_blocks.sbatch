#!/bin/bash
# GEMM v2 block-size experiment: why does the kernel stop at 41-49% of cuBLAS?
#
# Each configuration is the committed extensions/gemm_v2 kernel with ONLY its four
# tiling constants changed (TM x TN outputs per thread, BN tile width, BK tile
# depth). Nothing in the repository is modified: each variant is generated in /tmp.
# For every variant the job runs the supplied GEMM test (correctness on all cases,
# throughput against cuBLAS) and, if it passes, a whole-model timing. PyTorch and
# the assigned kernels are timed in the same job, so every number shares a session.
#
# From hw1-asr, on the head node:
#     sbatch extensions/run_gemm_blocks.sbatch
#     squeue -u "$USER"
#     cat extensions/results/gemm_blocks_JOBID/summary.txt
#SBATCH --job-name=mls-gemm-blocks
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=gemm-blocks-%j.out
#SBATCH --no-requeue

set -o pipefail
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"

OUT=extensions/results/gemm_blocks_${SLURM_JOB_ID}
WORK=/tmp/$USER/gemm_blocks_${SLURM_JOB_ID}
mkdir -p "$OUT" "$WORK"
BASE_FLAGS="-O3 -std=c++17 -lineinfo -diag-suppress 177"   # the Makefile's own flags

{
  date -Is; hostname; nvidia-smi -L
  nvidia-smi --query-gpu=name,uuid,compute_cap --format=csv
  echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "gemm_v2 source sha256: $(sha256sum extensions/gemm_v2/task3_gemm.cu | cut -d' ' -f1)"
} | tee "$OUT/00_environment.txt"

echo; echo "===== PyTorch baseline (start) ====="
./run.sh --cuda none --profile 2>&1 | tee "$OUT/e2e_pytorch_start.txt"
echo; echo "===== Assigned kernels (original GEMM) ====="
make -s test3 2>&1 | tee "$OUT/test_original.txt"
./run.sh --profile 2>&1 | tee "$OUT/e2e_original.txt"

# TM TN BN BK   (4 4 64 16 is the submitted GEMM v2)
CONFIGS="
4 4 64 16
2 2 64 16
4 8 64 16
8 4 64 16
8 8 64 16
8 8 128 16
4 4 128 16
4 4 64 8
4 4 64 32
"

echo "$CONFIGS" | while read -r TM TN BN BK; do
  [ -z "$TM" ] && continue
  name=tm${TM}_tn${TN}_bn${BN}_bk${BK}
  src=$WORK/$name/src
  build=$WORK/$name/build
  echo; echo "===== $name ====="
  mkdir -p "$src"
  cp -rL extensions/gemm_v2/. "$src/"          # -L: follow the symbolic links
  sed -i -e "s/^constexpr int BN = 64;/constexpr int BN = ${BN};/" \
         -e "s/^constexpr int BK = 16;/constexpr int BK = ${BK};/" \
         -e "s/^constexpr int TM = 4;/constexpr int TM = ${TM};/" \
         -e "s/^constexpr int TN = 4;/constexpr int TN = ${TN};/" "$src/task3_gemm.cu"
  {
    echo "config: TM=$TM TN=$TN BN=$BN BK=$BK"
    grep -n -E "^constexpr int (BN|BK|TM|TN) = " "$src/task3_gemm.cu"
    echo "lines differing from extensions/gemm_v2/task3_gemm.cu: $(diff extensions/gemm_v2/task3_gemm.cu "$src/task3_gemm.cu" | grep -c '^>')"
  } | tee "$OUT/test_${name}.txt"
  ok=1
  for pair in "BN ${BN}" "BK ${BK}" "TM ${TM}" "TN ${TN}"; do
    set -- $pair
    grep -q "^constexpr int $1 = $2;" "$src/task3_gemm.cu" || ok=0
  done
  if [ $ok -ne 1 ]; then
    echo "SKIPPED: the constants were not found in the source as expected" | tee -a "$OUT/test_${name}.txt"
    continue
  fi
  if make -s SRC="$src" BUILD="$build" NVCCFLAGS="$BASE_FLAGS" test3 >> "$OUT/test_${name}.txt" 2>&1; then
    tail -n 22 "$OUT/test_${name}.txt"
    make -s SRC="$src" BUILD="$build" NVCCFLAGS="$BASE_FLAGS" lib >> "$OUT/test_${name}.txt" 2>&1
    env ASR_LIB="$build/libasr_kernels.so" ./run.sh --profile 2>&1 | tee "$OUT/e2e_${name}.txt"
  else
    tail -n 30 "$OUT/test_${name}.txt"
    echo "BUILD OR TEST FAILED for $name: no whole-model run" | tee -a "$OUT/test_${name}.txt"
  fi
done

echo; echo "===== PyTorch baseline (end) ====="
./run.sh --cuda none --profile 2>&1 | tee "$OUT/e2e_pytorch_end.txt"

echo; echo "===== SUMMARY ====="
python extensions/summarise_gemm_blocks.py "$OUT" | tee "$OUT/summary.txt"
date -Is | tee -a "$OUT/00_environment.txt"
sed -i "s/$USER/UUN/g" "$OUT"/*.txt   # keep the saved logs free of the username
rm -rf "$WORK"

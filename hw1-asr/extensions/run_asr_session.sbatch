#!/bin/bash
# One-session ASR comparison: every backend timed on the same node, same job.
# From hw1-asr, on the head node:
#     sbatch extensions/run_asr_session.sbatch
#     squeue -u "$USER"                      # wait until the job is gone
#     ls extensions/results/session_JOBID/   # one log per step
#SBATCH --job-name=mls-asr-session
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=asr-session-%j.out
#SBATCH --no-requeue

set -eo pipefail
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"

OUT=extensions/results/session_${SLURM_JOB_ID}
mkdir -p "$OUT"
V2_BUILD=/tmp/$USER/build_v2_${SLURM_JOB_ID}
V2_LIB=$V2_BUILD/libasr_kernels.so
mkdir -p "$V2_BUILD"

{
  date -Is
  hostname
  nvidia-smi -L
  nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total --format=csv
  echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  git status --short csrc extensions/gemm_v2 2>/dev/null || true
} | tee "$OUT/00_environment.txt"

step () {  # $1 = log name, rest = command
  local name=$1; shift
  echo; echo "===== $name: $* ====="
  "$@" 2>&1 | tee "$OUT/$name.txt"
}

# Correctness of the assigned kernels (33 cases), then build GEMM v2 and test it.
step 01_make_test             make test
step 02_v2_build_and_test3    make SRC=extensions/gemm_v2 BUILD="$V2_BUILD" test3
make -s SRC=extensions/gemm_v2 BUILD="$V2_BUILD" lib

# Whole-model timings: one warm-up, three timed runs, then a profiled run.
step 03_pytorch_start         ./run.sh --cuda none --profile
step 04_rope_reduce           ./run.sh --cuda rope,reduce --profile
step 05_v1_all_custom         ./run.sh --profile
step 06_v1_pytorch_attention  ./run.sh --cuda rope,reduce,gemm --profile
step 07_v2_all_custom         env ASR_LIB="$V2_LIB" ./run.sh --profile
step 08_v2_pytorch_attention  env ASR_LIB="$V2_LIB" ./run.sh --cuda rope,reduce,gemm --profile
step 09_pytorch_end           ./run.sh --cuda none --profile

# Steps 05 vs 06 and 07 vs 08 switch only the attention kernel: their wall-clock
# difference is the attention comparison the profiles could not settle.
# Steps 03 and 09 bracket the session: their difference is the drift.
echo; echo "===== SUMMARY (ms per transcription) ====="
grep -H -E "ms per transcription|ms/transcription|mean" "$OUT"/0[3-9]_*.txt | tee "$OUT/10_summary.txt" || true
date -Is | tee -a "$OUT/00_environment.txt"

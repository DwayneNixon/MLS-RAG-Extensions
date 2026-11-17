#!/bin/bash
# Final validation of the five task files in csrc/, in one job on one GPU:
# GPU identity, clean `make test` (33 cases), `./run.sh --profile` (Status PASS),
# and racecheck on the GEMM test binary (small cases, checked against the CPU).
#
# From hw1-asr, on the head node:
#     sbatch extensions/run_verify_submission.sbatch
#     cat extensions/results/verify_submission_JOBID/summary.txt
#SBATCH --job-name=mls-verify
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=verify-%j.out
#SBATCH --no-requeue

set -o pipefail
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
[ -f csrc/task3_gemm.cu ] || { echo "Run from hw1-asr"; exit 1; }

OUT=extensions/results/verify_submission_${SLURM_JOB_ID}
mkdir -p "$OUT"
{
  date -Is; hostname; nvidia-smi -L
  echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  git status --short csrc 2>/dev/null
  for f in csrc/*.cu csrc/*.cuh; do echo "$f sha256: $(sha256sum "$f" | cut -d' ' -f1)"; done
} | tee "$OUT/00_environment.txt"

make clean
make test 2>&1 | tee "$OUT/01_make_test.txt"
./run.sh --profile 2>&1 | tee "$OUT/02_run_profile.txt"
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
compute-sanitizer --tool racecheck "build/sm_${SM}/test_task3_gemm" small 2>&1 | tee "$OUT/03_racecheck_task3.txt"

{
  echo "Verification of csrc/ ($(date -Is))"
  echo "test cases PASS: $(grep -cE '\s{2,}PASS\s' "$OUT/01_make_test.txt"), FAIL: $(grep -cE '\s{2,}FAIL\s' "$OUT/01_make_test.txt")"
  grep -E "ALL PASS|FAILED" "$OUT/01_make_test.txt"
  grep -E "^Time:|^Transcription:|^Status:|^Peak GPU" "$OUT/02_run_profile.txt"
  grep -E "RACECHECK SUMMARY|ERROR SUMMARY" "$OUT/03_racecheck_task3.txt"
} | tee "$OUT/summary.txt"
sed -i "s/$USER/UUN/g" "$OUT"/*.txt

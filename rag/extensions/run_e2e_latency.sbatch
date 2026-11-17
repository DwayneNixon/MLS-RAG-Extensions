#!/bin/bash
# OPTIONAL (item 6): end-to-end latency per stage, three ASR backends, one GPU job.
# Needs both environments: the RAG venv (runs the pipeline) and the mls-asr conda env
# (runs the ASR worker on the GPU). Compute nodes reach ELM.
#
# From the repository root, on the head node:
#     RAG_VENV=~/mls-rag-env sbatch rag/extensions/run_e2e_latency.sbatch
#SBATCH --job-name=mls-e2e
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=e2e-latency-%j.out
#SBATCH --no-requeue

set -eo pipefail
export PATH=/opt/cuda-12.8.0/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
[ -f rag/extensions/e2e_latency.py ] || { echo "Run from the repository root"; exit 1; }
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
export RAG_ASR_PYTHON=$CONDA_BASE/envs/mls-asr/bin/python
export RAG_ASR_ROOT=$PWD/hw1-asr
source "${RAG_VENV:-$HOME/mls-rag-env}/bin/activate"

OUT=rag/results/phase5/e2e_latency_${SLURM_JOB_ID}
mkdir -p "$OUT"
V2_BUILD=/tmp/$USER/e2e_v2_${SLURM_JOB_ID}
( cd hw1-asr && make -s lib && make -s SRC=extensions/gemm_v2 BUILD="$V2_BUILD" lib )
{ date -Is; hostname; nvidia-smi -L; echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"; } \
  | tee "$OUT/00_environment.txt"

ASR_CUDA=none python rag/extensions/e2e_latency.py --label pytorch  --output "$OUT/pytorch.jsonl"  | tee "$OUT/pytorch.txt"
ASR_CUDA=all  python rag/extensions/e2e_latency.py --label assigned --output "$OUT/assigned.jsonl" | tee "$OUT/assigned.txt"
ASR_CUDA=all ASR_LIB="$V2_BUILD/libasr_kernels.so" \
              python rag/extensions/e2e_latency.py --label v2       --output "$OUT/v2.jsonl"       | tee "$OUT/v2.txt"
date -Is | tee -a "$OUT/00_environment.txt"
sed -i "s/$USER/UUN/g" "$OUT"/*.txt
rm -rf "$V2_BUILD"

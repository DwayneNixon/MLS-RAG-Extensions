#!/bin/bash
#SBATCH --job-name=mls-rerank-matched
#SBATCH --partition=Teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=damnii[07-12]
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=rag/results/matched_rerank_damnii_%j.log
#SBATCH --no-requeue

set -euo pipefail

cd /home/s3024630/edin-mls-26-autumn

case "$(hostname -s)" in
    damnii07|damnii08|damnii09|damnii10|damnii11|damnii12)
        ;;
    *)
        echo "ERROR: Allocation is not on an authorised damnii node."
        exit 1
        ;;
esac

PYTHON="/home/s3024630/mls-rag-env/bin/python"
OUTPUT="rag/results/reranking_matched_damnii_${SLURM_JOB_ID}.json"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "===== EXPERIMENT ENVIRONMENT ====="
date -Is
hostname
echo "Slurm job: $SLURM_JOB_ID"
echo "Python: $PYTHON"

"$PYTHON" -c \
    'import torch, sentence_transformers; print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())'

nvidia-smi -L

echo
echo "===== MATCHED BENCHMARK ====="

"$PYTHON" rag/extensions/benchmark_reranking_matched.py \
    --repeats 3 \
    --warmup 5 \
    --output "$OUTPUT"

echo
echo "===== COMPLETED ====="
date -Is
echo "Results: $OUTPUT"

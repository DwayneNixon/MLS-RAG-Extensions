#!/bin/bash
# Answer-length follow-up (ANALYSIS_PLAN_answer_length.md): four interleaved arms on
# Development in one session, then the judge on k5_cap150 and k5_short.
# Run from the repository root, inside tmux, on a machine that reaches ELM (the same
# set-up as run_practice_depth.sh; no GPU needed).
#
#     bash rag/extensions/run_answer_length.sh          # full run (about 1.5-2 hours)
#     SMOKE=1 bash rag/extensions/run_answer_length.sh  # 3 queries, separate folder
#
# Safe to re-run after a crash: finished answers and judgements are kept.
set -euo pipefail

PY=${PYTHON:-python}
CACHE=rag/.cache/dev_transcripts.jsonl
PLAN=rag/extensions/ANALYSIS_PLAN_answer_length.md
OUT=rag/results/phase5/answer_length
LIMIT_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
  OUT=rag/results/phase5/answer_length_smoke
  LIMIT_ARGS=(--limit 3)
fi
mkdir -p "$OUT/logs"
LOG="$OUT/logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "===== PRE-FLIGHT ====="
date -Is; hostname
[ -f rag/extensions/generate_length_arms.py ] || { echo "Run this from the repository root."; exit 1; }
[ -f "$CACHE" ] || { echo "Missing $CACHE"; exit 1; }
echo "Cached dev transcripts: $(grep -c . "$CACHE") (expected 125)"
[ -f rag/.env ] || { echo "Missing rag/.env (ELM key)"; exit 1; }
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
if [ "${SMOKE:-0}" != "1" ]; then
  if ! git ls-files --error-unmatch "$PLAN" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$PLAN"; then
    echo "ERROR: commit $PLAN before launching, so its timestamp precedes the results."
    exit 1
  fi
  echo "Plan committed at: $(git log -1 --format=%cI -- "$PLAN")"
fi

echo; echo "===== GENERATE (interleaved) $(date -Is) ====="
tries=0
until "$PY" rag/extensions/generate_length_arms.py --split dev --transcript-cache "$CACHE" \
        --out-dir "$OUT" "${LIMIT_ARGS[@]}"; do
  tries=$((tries + 1))
  [ $tries -ge 5 ] && { echo "Generation failed 5 times; stopping."; exit 1; }
  echo "Generation stopped; resuming in 120s (attempt $tries/5)"; sleep 120
done

for arm in k5_short k5_cap150; do
  echo; echo "===== JUDGE $arm $(date -Is) ====="
  if [ "${SMOKE:-0}" = "1" ]; then
    "$PY" rag/scripts/evaluate_answers.py --candidates "$OUT/$arm/answers.jsonl" \
      --split dev --out-dir "$OUT/$arm/judge" --repeats 1 --api-retries 8 --allow-subset
  else
    "$PY" rag/extensions/judge_resumable.py --candidates "$OUT/$arm/answers.jsonl" \
      --split dev --out-dir "$OUT/$arm/judge" --api-retries 8 --max-excluded 20
  fi
done

echo; echo "===== ANALYSIS $(date -Is) ====="
"$PY" rag/extensions/analyse_answer_length.py "$OUT" | tee "$OUT/analysis_output.txt"
echo "===== COMPLETED $(date -Is) ====="

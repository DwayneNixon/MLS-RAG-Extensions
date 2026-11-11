#!/bin/bash
# Practice-split depth replication: dense k=5, k=2, then k=20, one generation each.
# Run from the repository root, inside tmux, in the same Python environment and on
# the same machine you used for run_phase3.sh (it needs to reach the ELM endpoint;
# no GPU is needed: the query encoder ran on CPU for the Development runs too).
#
#     bash rag/extensions/run_practice_depth.sh              # k5, k2, k20
#     ARMS="k5 k2" bash rag/extensions/run_practice_depth.sh # primary pair only
#     SMOKE=1 bash rag/extensions/run_practice_depth.sh      # 3 queries, separate folder
#
# Safe to re-run after a crash: finished answers and judgements are kept.
set -euo pipefail

PY=${PYTHON:-python}
ARMS=${ARMS:-"k5 k2 k20"}
SPLIT=practice
CACHE=rag/.cache/practice_transcripts.jsonl
PLAN=rag/extensions/ANALYSIS_PLAN_practice_depth.md
OUT=rag/results/practice_depth
LIMIT_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
  OUT=rag/results/practice_depth_smoke
  LIMIT_ARGS=(--limit 3)
fi
mkdir -p "$OUT/logs"
LOG="$OUT/logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

k_of () { case "$1" in k2) echo 2;; k5) echo 5;; k20) echo 20;; *) echo "unknown arm $1" >&2; exit 1;; esac; }

echo "===== PRE-FLIGHT ====="
date -Is; hostname
[ -f rag/extensions/generate_answers_cached.py ] || { echo "Run this from the repository root."; exit 1; }
[ -f "$CACHE" ] || { echo "Missing $CACHE"; exit 1; }
echo "Cached practice transcripts: $(grep -c . "$CACHE") (expected 350)"
[ -f rag/.env ] || { echo "Missing rag/.env (ELM key)"; exit 1; }
grep -E '^ELM_(MODEL|BASE_URL|JUDGE_MODEL)=' rag/.env || true
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
if [ "${SMOKE:-0}" != "1" ]; then
  if ! git ls-files --error-unmatch "$PLAN" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$PLAN"; then
    echo "ERROR: commit $PLAN before launching, so its timestamp precedes the results."
    exit 1
  fi
  echo "Plan committed at: $(git log -1 --format=%cI -- "$PLAN")"
fi

generate () {  # $1 = arm
  local arm=$1 k; k=$(k_of "$arm")
  echo; echo "===== GENERATE $arm (k=$k) $(date -Is) ====="
  local tries=0
  until "$PY" rag/extensions/generate_answers_resumable.py --split $SPLIT \
          --transcript-cache "$CACHE" --config "$arm" --k "$k" \
          --output "$OUT/$arm/answers.jsonl" "${LIMIT_ARGS[@]}"; do
    tries=$((tries + 1))
    [ $tries -ge 5 ] && { echo "Generation for $arm failed 5 times; stopping."; exit 1; }
    echo "Generation for $arm stopped; resuming in 120s (attempt $tries/5)"; sleep 120
  done
}

judge () {  # $1 = arm
  local arm=$1
  echo; echo "===== JUDGE $arm $(date -Is) ====="
  if [ "${SMOKE:-0}" = "1" ]; then
    "$PY" rag/scripts/evaluate_answers.py --candidates "$OUT/$arm/answers.jsonl" \
      --split $SPLIT --out-dir "$OUT/$arm/judge" --repeats 1 --api-retries 8 --allow-subset
  else
    "$PY" rag/extensions/judge_resumable.py --candidates "$OUT/$arm/answers.jsonl" \
      --split $SPLIT --out-dir "$OUT/$arm/judge" --api-retries 8
  fi
}

# Primary pair first (k5 then k2), each fully judged before k20 starts, so the
# primary comparison is complete even if the night ends early.
for arm in $ARMS; do
  mkdir -p "$OUT/$arm"
  generate "$arm"
  judge "$arm"
done

echo; echo "===== ANALYSIS $(date -Is) ====="
if [ "${SMOKE:-0}" = "1" ]; then
  echo "Smoke run complete: check the three answers and scores above, then launch the full run."
else
  "$PY" rag/extensions/analyse_practice_depth.py | tee "$OUT/analysis_output.txt"
fi
echo "===== COMPLETED $(date -Is) ====="

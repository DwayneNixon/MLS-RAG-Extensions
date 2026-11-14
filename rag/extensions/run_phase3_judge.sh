#!/bin/bash
set -e
for c in k5b; do
  r=1; [ "$c" = "k5" ] && r=3
  echo "=== $c (repeats $r) ==="
  python rag/scripts/evaluate_answers.py --candidates rag/results/phase3/$c/answers.jsonl \
    --split dev --out-dir rag/results/phase3/$c/judge --repeats $r --api-retries 5
done

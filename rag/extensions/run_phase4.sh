#!/bin/bash
set -e
OUT=rag/results/phase4
CACHE=rag/.cache/dev_transcripts.jsonl
mkdir -p $OUT/logs
for c in gold_first gold_last; do
  mkdir -p $OUT/$c
  python rag/extensions/generate_answers_cached.py --split dev --transcript-cache $CACHE \
    --config $c --k 5 --evidence $c --output $OUT/$c/answers.jsonl
done
for c in gold_first gold_last; do
  echo "=== judging $c"
  python rag/scripts/evaluate_answers.py --candidates $OUT/$c/answers.jsonl \
    --split dev --out-dir $OUT/$c/judge --repeats 1 --api-retries 8
done

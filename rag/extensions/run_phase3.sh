#!/bin/bash
set -e
OUT=rag/results/phase3
CACHE=rag/.cache/dev_transcripts.jsonl
mkdir -p $OUT/logs

run () {   # $1 = label, $2 = k
  mkdir -p $OUT/$1
  python rag/extensions/generate_answers_cached.py --split dev \
    --transcript-cache $CACHE --config $1 --k $2 \
    --output $OUT/$1/answers.jsonl
}

run k5  5
run k2  2
run k10 10
run k20 20
run k5b 5     
"""Generate ELM answers from previously evaluated reranker rankings.

Reuses the shared generator and the original context builder.
Does not rerun ASR, query encoding, dense retrieval or reranking.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.context_builder import SimpleContextBuilder
from extensions.generate_answers_cached import (
    ELMGeneratorWithUsage,
    load_cache,
)

MINILM_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def load_rankings(path):
    rows = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            qid = row["query_id"]

            if qid in rows:
                raise ValueError(f"Duplicate query ID: {qid}")

            rows[qid] = row

    return rows


def validate_source(row, split, config, transcript):
    if row["split"] != split:
        raise ValueError("Ranking split mismatch")

    if row["transcript"] != transcript:
        raise ValueError(
            f"Transcript mismatch for {row['query_id']}"
        )

    if config == "hybrid_alpha095":
        if row["config"] != "hybrid":
            raise ValueError("Wrong hybrid source")

        if abs(float(row["alpha"]) - 0.95) > 1e-12:
            raise ValueError("Hybrid alpha is not 0.95")

    elif config == "minilm_top100_to5":
        if row["config"] != "minilm_top100_to5":
            raise ValueError("Wrong MiniLM source")

        if row["model_revision"] != MINILM_REVISION:
            raise ValueError("Wrong MiniLM checkpoint")

        if row["batch_size"] != 16:
            raise ValueError("Unexpected MiniLM batch size")

        if row["max_length"] != 512:
            raise ValueError("Unexpected MiniLM max length")

    dense = row["dense_record_ids"]
    reranked = row["reranked_record_ids"]
    selected = row["retrieved_record_ids"]

    if len(dense) != 100 or len(set(dense)) != 100:
        raise ValueError("Invalid dense candidate pool")

    if len(reranked) != 100:
        raise ValueError("Invalid reranked pool")

    if set(reranked) != set(dense):
        raise ValueError("Reranking changed candidate membership")

    if len(selected) != 5 or selected != reranked[:5]:
        raise ValueError("Selected top five do not match ranking")

    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        "--split",
        choices=["practice", "dev"],
        required=True,
    )
    ap.add_argument(
        "--config",
        choices=["hybrid_alpha095", "minilm_top100_to5"],
        required=True,
    )
    ap.add_argument("--rankings", type=Path, required=True)
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5)

    # Safe default: validation only. No ELM requests unless explicit.
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually call ELM and save generated answers",
    )

    args = ap.parse_args()

    if args.limit < 1:
        ap.error("--limit must be positive")

    if not args.rankings.is_file():
        ap.error("Ranking file does not exist")

    if not args.transcript_cache.is_file():
        ap.error("Transcript cache does not exist")

    if args.output.exists():
        ap.error("Refusing to overwrite existing results")

    data = CourseData(ROOT / "data/rag_corpus")
    items = data.audio_items(args.split)
    cache = load_cache(args.transcript_cache)
    rankings = load_rankings(args.rankings)

    expected_ids = [item["query_id"] for item in items]

    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Duplicate dataset query IDs")

    if set(rankings) != set(expected_ids):
        raise ValueError(
            "Ranking file does not cover the exact dataset split"
        )

    if any(qid not in cache for qid in expected_ids):
        raise ValueError("Transcript cache has missing queries")

    if args.limit > len(items):
        ap.error("--limit exceeds dataset size")

    # Validate the full ranking file, not just the requested subset.
    for item in items:
        qid = item["query_id"]
        validate_source(
            rankings[qid],
            args.split,
            args.config,
            cache[qid],
        )

    builder = SimpleContextBuilder()

    print("===== SAVED-RANKING ELM GENERATION =====")
    print("Split:", args.split)
    print("Config:", args.config)
    print("Ranking entries validated:", len(rankings))
    print("Requested answers:", args.limit)
    print("Mode:", "EXECUTE" if args.execute else "DRY RUN")
    print()

    selected_items = items[:args.limit]
    prepared = []

    for item in selected_items:
        qid = item["query_id"]
        transcript = cache[qid]
        source = rankings[qid]

        record_ids = source["retrieved_record_ids"]

        built = builder.build(
            transcript,
            record_ids,
            data,
        )

        prompt = built["prompt"]

        prepared.append(
            (qid, transcript, source, record_ids, prompt)
        )

        print(
            qid,
            "documents=", len(record_ids),
            "prompt_chars=", len(prompt),
            "first_record=", record_ids[0],
            flush=True,
        )

    if not args.execute:
        print("\nDRY RUN: PASS")
        print("No ELM requests made.")
        print("No output file created.")
        return

    # Instantiate the exact shared generator implementation.
    generator = ELMGeneratorWithUsage()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("x", encoding="utf-8") as f:
        for i, (
            qid, transcript, source, record_ids, prompt
        ) in enumerate(prepared, start=1):

            gen = generator.generate(prompt)
            source_timing = source.get("timing", {})

            row = {
                "query_id": qid,
                "split": args.split,
                "config": args.config,
                "k": 5,
                "transcript": transcript,
                "answer": gen["answer"],
                "retrieved_record_ids": record_ids,
                "prompt_chars": len(prompt),
                "prompt_tokens": gen["prompt_tokens"],
                "answer_chars": len(gen["answer"]),
                "answer_tokens": gen["answer_tokens"],
                "finish_reason": gen["finish_reason"],
                "timing": {
                    "embedding_ms": source_timing.get(
                        "embedding_ms"
                    ),
                    "retrieval_ms": source_timing.get(
                        "retrieval_ms"
                    ),
                    "rerank_ms": source_timing.get(
                        "rerank_ms"
                    ),
                    "generation_ms": gen["latency_ms"],
                },
                "ranking_source": str(args.rankings),
                "timing_note": (
                    "Embedding/retrieval/rerank timings, "
                    "where present, are from earlier experiments; "
                    "only generation_ms is measured in this run."
                ),
            }

            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            f.flush()

            print(
                f"{i}/{len(prepared)} {qid} "
                f"prompt_tokens={gen['prompt_tokens']} "
                f"answer_tokens={gen['answer_tokens']} "
                f"generation_ms={gen['latency_ms']:.2f}",
                flush=True,
            )

    print("\nAnswers saved:", args.output)
    print("SAVED-RANKING ELM GENERATION: PASS")


if __name__ == "__main__":
    main()

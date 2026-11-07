"""Resumable MiniLM Development answer generation.

Uses saved MiniLM rankings and the original ELM generator.
Dry run by default. Use --execute for API requests.
"""

import argparse
import hashlib
import json
import os
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

from extensions.generate_answers_reranked import (
    load_rankings,
    validate_source,
)


RESULTS = ROOT / "results"

RANKINGS = RESULTS / "minilm_dev_full125.jsonl"
TRANSCRIPTS = ROOT / ".cache/dev_transcripts.jsonl"

CHECKPOINT = RESULTS / "minilm_dev_answers_checkpoint.jsonl"
FINAL = RESULTS / "minilm_dev_answers_full125.jsonl"

CONFIG = "minilm_top100_to5"


def sha256(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually generate answers through ELM",
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=10,
        help="Maximum new answers per execution",
    )

    args = parser.parse_args()

    if args.max_new < 1:
        parser.error("--max-new must be positive")

    if FINAL.exists():
        raise FileExistsError(
            f"Final file already exists: {FINAL}. "
            "Refusing to overwrite completed results."
        )

    # Load original Development data.
    data = CourseData(ROOT / "data/rag_corpus")
    items = data.audio_items("dev")

    transcripts = load_cache(TRANSCRIPTS)
    rankings = load_rankings(RANKINGS)

    ids = [item["query_id"] for item in items]

    assert len(ids) == 125
    assert len(set(ids)) == 125
    assert set(rankings) == set(ids)
    assert all(qid in transcripts for qid in ids)

    # Reconstruct and validate all MiniLM prompts.
    builder = SimpleContextBuilder()
    prepared = {}

    for item in items:
        qid = item["query_id"]

        transcript = transcripts[qid]
        source = rankings[qid]

        selected = validate_source(
            source,
            "dev",
            CONFIG,
            transcript,
        )

        built = builder.build(
            transcript,
            selected,
            data,
        )

        prompt = built["prompt"]

        prepared[qid] = {
            "transcript": transcript,
            "source": source,
            "selected": selected,
            "prompt": prompt,
            "prompt_sha256": sha256(prompt),
        }

    # Recover previously generated answers.
    saved_rows = read_jsonl(CHECKPOINT)
    saved = {}

    for row in saved_rows:
        qid = row["query_id"]

        if qid in saved:
            raise ValueError(
                f"Duplicate checkpoint query: {qid}"
            )

        if qid not in prepared:
            raise ValueError(
                f"Unknown checkpoint query: {qid}"
            )

        expected = prepared[qid]

        assert row["split"] == "dev"
        assert row["config"] == CONFIG
        assert row["k"] == 5

        assert (
            row["transcript"]
            == expected["transcript"]
        )

        assert (
            row["retrieved_record_ids"]
            == expected["selected"]
        )

        assert (
            row["prompt_sha256"]
            == expected["prompt_sha256"]
        )

        assert (
            row["prompt_chars"]
            == len(expected["prompt"])
        )

        assert (
            isinstance(row["answer"], str)
            and row["answer"].strip()
        )

        saved[qid] = row

    missing = [
        qid for qid in ids
        if qid not in saved
    ]

    print(
        "\n===== RESUMABLE MINILM DEV GENERATION =====",
        flush=True,
    )

    print("Development questions:", len(ids))
    print("Valid saved answers:", len(saved))
    print("Remaining answers:", len(missing))
    print("Maximum new answers:", args.max_new)
    print("Mode:", "EXECUTE" if args.execute else "DRY RUN")
    print("Checkpoint:", CHECKPOINT)
    print("Final output:", FINAL)

    if missing:
        print("First missing queries:", missing[:5])

    # No API calls unless --execute is provided.
    if not args.execute:
        print("\nDRY RUN: PASS")
        print("No ELM requests made.")
        print("No result files modified.")
        return

    if missing:
        generator = ELMGeneratorWithUsage()

        CHECKPOINT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with CHECKPOINT.open("a", encoding="utf-8") as f:

            for qid in missing[:args.max_new]:
                info = prepared[qid]

                gen = generator.generate(info["prompt"])

                answer = gen["answer"]

                if (
                    not isinstance(answer, str)
                    or not answer.strip()
                ):
                    raise ValueError(
                        f"Empty generated answer: {qid}"
                    )

                timing = info["source"].get("timing", {})

                row = {
                    "query_id": qid,
                    "split": "dev",
                    "config": CONFIG,
                    "k": 5,
                    "transcript": info["transcript"],
                    "answer": answer,
                    "retrieved_record_ids": info["selected"],
                    "prompt_chars": len(info["prompt"]),
                    "prompt_sha256": info["prompt_sha256"],
                    "prompt_tokens": gen["prompt_tokens"],
                    "answer_chars": len(answer),
                    "answer_tokens": gen["answer_tokens"],
                    "finish_reason": gen["finish_reason"],
                    "timing": {
                        "embedding_ms": timing.get(
                            "embedding_ms"
                        ),
                        "retrieval_ms": timing.get(
                            "retrieval_ms"
                        ),
                        "rerank_ms": timing.get(
                            "rerank_ms"
                        ),
                        "generation_ms": gen["latency_ms"],
                    },
                    "ranking_source": str(RANKINGS),
                    "answer_origin": "generated_new",
                    "timing_note": (
                        "Only generation_ms is measured "
                        "in this run. Other timings come "
                        "from saved retrieval results."
                    ),
                }

                # Save each answer immediately.
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    ) + "\n"
                )

                f.flush()
                os.fsync(f.fileno())

                saved[qid] = row

                print(
                    f"Saved {len(saved)}/125 "
                    f"{qid} "
                    f"generation_ms="
                    f"{gen['latency_ms']:.2f}",
                    flush=True,
                )

    remaining = [
        qid for qid in ids
        if qid not in saved
    ]

    print("\nTotal saved:", len(saved))
    print("Still missing:", len(remaining))

    if remaining:
        print(
            "CHECKPOINT SAVED — "
            "resume with --execute"
        )
        return

    # Construct final results only after 125 valid answers.
    # Never overwrite an existing final file.
    temp = FINAL.with_name(
        FINAL.name + f".tmp.{os.getpid()}"
    )

    with temp.open("x", encoding="utf-8") as f:
        for qid in ids:
            f.write(
                json.dumps(
                    saved[qid],
                    ensure_ascii=False,
                ) + "\n"
            )

        f.flush()
        os.fsync(f.fileno())

    try:
        os.link(temp, FINAL)
    finally:
        temp.unlink(missing_ok=True)

    print("\n===== FINAL MINILM DATASET =====")
    print("Answers:", len(saved))
    print("Coverage: 125/125")
    print("Saved:", FINAL)
    print("MINILM GENERATION: PASS")


if __name__ == "__main__":
    main()

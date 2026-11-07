"""Resumable, provenance-aware hybrid Development answer generation."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.context_builder import SimpleContextBuilder
from extensions.generate_answers_cached import (
    ELMGeneratorWithUsage,
    load_cache,
)

RESULTS = ROOT / "results"

RANKINGS = RESULTS / "hybrid_dev_alpha095.jsonl"
BASELINE = RESULTS / "phase3_import/k5b/answers.jsonl"
SMOKE = RESULTS / "hybrid_dev_answers_smoke5.jsonl"
TRANSCRIPTS = ROOT / ".cache/dev_transcripts.jsonl"

PROGRESS = RESULTS / "hybrid_dev_new_answers.jsonl"
FINAL = RESULTS / "hybrid_dev_answers_full125.jsonl"


def load_rows(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}

    rows = {}

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            qid = row["query_id"]

            if qid in rows:
                raise ValueError(
                    f"Duplicate query {qid} in {path}"
                )

            rows[qid] = row

    return rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        "--execute",
        action="store_true",
        help="Make real ELM requests; default is dry run",
    )

    ap.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Maximum new ELM requests during this run",
    )

    args = ap.parse_args()

    if args.max_new is not None and args.max_new < 1:
        ap.error("--max-new must be positive")

    if FINAL.exists():
        ap.error(
            f"Final output already exists: {FINAL}. "
            "Refusing to overwrite."
        )

    data = CourseData(ROOT / "data/rag_corpus")
    builder = SimpleContextBuilder()

    items = data.audio_items("dev")
    qids = [item["query_id"] for item in items]

    require(len(qids) == 125, "Expected 125 Dev queries")
    require(len(set(qids)) == 125, "Duplicate dataset IDs")

    cache = load_cache(TRANSCRIPTS)
    hybrid = load_rows(RANKINGS)
    baseline = load_rows(BASELINE)
    smoke = load_rows(SMOKE)
    progress = load_rows(PROGRESS, required=False)

    expected = set(qids)

    require(set(hybrid) == expected, "Hybrid coverage mismatch")
    require(set(baseline) == expected, "Baseline coverage mismatch")
    require(set(smoke).issubset(expected), "Invalid smoke IDs")
    require(len(smoke) == 5, "Expected five smoke answers")
    require(
        expected.issubset(cache),
        "Missing cached transcripts"
    )

    prompts = {}
    prompt_hashes = {}

    reuse_dense = {}
    reuse_smoke = {}

    for qid in qids:
        h = hybrid[qid]
        b = baseline[qid]
        transcript = cache[qid]

        require(
            h["split"] == "dev"
            and h["config"] == "hybrid"
            and abs(float(h["alpha"]) - 0.95) < 1e-12,
            f"Wrong hybrid configuration: {qid}"
        )

        require(
            h["transcript"] == transcript,
            f"Hybrid transcript mismatch: {qid}"
        )

        dense = h["dense_record_ids"]
        reranked = h["reranked_record_ids"]
        selected = h["retrieved_record_ids"]

        require(
            len(dense) == len(set(dense)) == 100,
            f"Invalid dense candidates: {qid}"
        )
        require(
            len(reranked) == len(set(reranked)) == 100,
            f"Invalid reranked candidates: {qid}"
        )
        require(
            set(dense) == set(reranked),
            f"Candidate pool changed: {qid}"
        )
        require(
            len(selected) == 5
            and selected == reranked[:5],
            f"Incorrect hybrid top five: {qid}"
        )

        require(
            b["split"] == "dev"
            and b["config"] == "k5b"
            and b["k"] == 5
            and b["transcript"] == transcript,
            f"Baseline configuration mismatch: {qid}"
        )

        # Reconstruct the baseline prompt independently.
        baseline_prompt = builder.build(
            transcript,
            b["retrieved_record_ids"],
            data,
        )["prompt"]

        require(
            len(baseline_prompt) == b["prompt_chars"],
            f"Baseline prompt length mismatch: {qid}"
        )

        # Reconstruct the new hybrid prompt.
        hybrid_prompt = builder.build(
            transcript,
            selected,
            data,
        )["prompt"]

        prompts[qid] = hybrid_prompt
        prompt_hashes[qid] = hashlib.sha256(
            hybrid_prompt.encode("utf-8")
        ).hexdigest()

        # Exact reconstructed prompt equality is stronger
        # than comparing document IDs or prompt lengths.
        if baseline_prompt == hybrid_prompt:
            reuse_dense[qid] = b

        elif qid in smoke:
            s = smoke[qid]

            require(
                s["split"] == "dev"
                and s["config"] == "hybrid_alpha095"
                and s["k"] == 5
                and s["transcript"] == transcript
                and s["retrieved_record_ids"] == selected
                and s["prompt_chars"] == len(hybrid_prompt),
                f"Smoke-answer mismatch: {qid}"
            )

            reuse_smoke[qid] = s

    # Validate every smoke-test record, including those
    # superseded by an identical-prompt baseline answer.
    for qid, s in smoke.items():
        require(
            s["split"] == "dev"
            and s["config"] == "hybrid_alpha095"
            and s["k"] == 5
            and s["transcript"] == cache[qid]
            and s["retrieved_record_ids"]
                == hybrid[qid]["retrieved_record_ids"]
            and s["prompt_chars"] == len(prompts[qid]),
            f"Invalid saved smoke answer: {qid}"
        )

    def build_answer(qid, source, origin, source_file):
        answer = source["answer"]

        require(
            isinstance(answer, str),
            f"Invalid answer type: {qid}"
        )

        timing = source.get("timing") or {}
        h = hybrid[qid]

        return {
            "query_id": qid,
            "split": "dev",
            "config": "hybrid_alpha095",
            "k": 5,
            "transcript": cache[qid],
            "answer": answer,
            "retrieved_record_ids": h["retrieved_record_ids"],
            "prompt_chars": len(prompts[qid]),
            "prompt_tokens": source.get("prompt_tokens"),
            "answer_chars": len(answer),
            "answer_tokens": source.get("answer_tokens"),
            "finish_reason": source.get("finish_reason"),
            "timing": {
                "embedding_ms": None,
                "retrieval_ms": None,
                "rerank_ms": (
                    (h.get("timing") or {}).get("rerank_ms")
                ),
                "generation_ms": timing.get("generation_ms"),
            },
            "answer_origin": origin,
            "source_answer_file": source_file,
            "prompt_sha256": prompt_hashes[qid],
            "timing_note": (
                "Generation latency comes from the answer's "
                "original run. Rerank latency comes from the "
                "saved retrieval evaluation."
            ),
        }

    new_needed = [
        qid for qid in qids
        if qid not in reuse_dense
        and qid not in reuse_smoke
    ]

    require(
        not (
            set(reuse_dense)
            & set(reuse_smoke)
        ),
        "Overlapping answer sources"
    )

    require(
        set(progress).issubset(set(new_needed)),
        "Progress contains an unexpected query"
    )

    # Fail closed if previously checkpointed answers
    # don't match this exact experiment.
    for qid, row in progress.items():
        require(
            row.get("answer_origin") == "generated_new"
            and row.get("config") == "hybrid_alpha095"
            and row.get("split") == "dev"
            and row.get("k") == 5
            and row.get("transcript") == cache[qid]
            and row.get("retrieved_record_ids")
                == hybrid[qid]["retrieved_record_ids"]
            and row.get("prompt_sha256")
                == prompt_hashes[qid]
            and row.get("prompt_chars")
                == len(prompts[qid])
            and isinstance(row.get("answer"), str),
            f"Saved progress mismatch: {qid}"
        )

    remaining = [
        qid for qid in new_needed
        if qid not in progress
    ]

    print("===== SELECTIVE HYBRID DEV GENERATION =====")
    print("Dev queries:", len(qids))
    print("Reuse dense k5b:", len(reuse_dense))
    print("Reuse hybrid smoke:", len(reuse_smoke))
    print("New generations required:", len(new_needed))
    print("New generations already saved:", len(progress))
    print("Still missing:", len(remaining))
    print("Mode:", "EXECUTE" if args.execute else "DRY RUN")

    print("\nFirst missing query IDs:")
    for qid in remaining[:8]:
        print(" ", qid)

    if not args.execute:
        print("\nDRY RUN: PASS")
        print("No ELM calls. No files modified.")
        return

    if remaining:
        generator = ELMGeneratorWithUsage()

        todo = remaining
        if args.max_new is not None:
            todo = todo[:args.max_new]

        PROGRESS.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with PROGRESS.open("a", encoding="utf-8") as f:
            for index, qid in enumerate(todo, start=1):
                gen = generator.generate(prompts[qid])

                source = {
                    "answer": gen["answer"],
                    "prompt_tokens": gen["prompt_tokens"],
                    "answer_tokens": gen["answer_tokens"],
                    "finish_reason": gen["finish_reason"],
                    "timing": {
                        "generation_ms": gen["latency_ms"]
                    },
                }

                row = build_answer(
                    qid,
                    source,
                    "generated_new",
                    str(PROGRESS.relative_to(ROOT)),
                )

                # Persist each successful response immediately.
                f.write(
                    json.dumps(row, ensure_ascii=False)
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())

                progress[qid] = row

                print(
                    f"{index}/{len(todo)} {qid} "
                    f"saved, generation_ms="
                    f"{gen['latency_ms']:.2f}",
                    flush=True,
                )

    missing = [
        qid for qid in new_needed
        if qid not in progress
    ]

    if missing:
        print("\nCHECKPOINT SAVED")
        print("Remaining new answers:", len(missing))
        print("Rerun with --execute to resume.")
        print("Final output not yet created.")
        return

    # Assemble in original dataset order.
    assembled = []

    for qid in qids:
        if qid in reuse_dense:
            row = build_answer(
                qid,
                reuse_dense[qid],
                "reused_dense_k5b_identical_prompt",
                str(BASELINE.relative_to(ROOT)),
            )
        elif qid in reuse_smoke:
            row = build_answer(
                qid,
                reuse_smoke[qid],
                "reused_hybrid_smoke",
                str(SMOKE.relative_to(ROOT)),
            )
        else:
            row = progress[qid]

        assembled.append(row)

    require(
        len(assembled) == 125
        and len({r["query_id"] for r in assembled}) == 125,
        "Final assembly incomplete"
    )

    # Stage a complete file, then publish without
    # overwriting any existing final result.
    FINAL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=FINAL.parent,
            prefix=".hybrid_dev_full_",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)

            for row in assembled:
                f.write(
                    json.dumps(row, ensure_ascii=False)
                    + "\n"
                )

            f.flush()
            os.fsync(f.fileno())

        os.link(temporary, FINAL)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    print("\n===== COMPLETE =====")
    print("Final answers:", len(assembled))
    print("Saved:", FINAL)
    print("SELECTIVE GENERATION: PASS")


if __name__ == "__main__":
    main()

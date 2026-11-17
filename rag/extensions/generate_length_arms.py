"""Answer-length follow-up: four arms generated in ONE session, interleaved by query.

Arms (see ANALYSIS_PLAN_answer_length.md):
    k5        dense top 5, unchanged pipeline (same-session latency reference)
    k2        dense top 2, unchanged pipeline
    k5_cap150 dense top 5, answer capped at max_tokens=150
    k5_short  dense top 5, one line appended to the user prompt:
              "Answer in at most 100 words."

For every query the four arms run back to back, in an order that rotates from
query to query, so slow periods of the shared ELM endpoint fall on all arms
alike. Everything else (encoder, retriever, context builder, system prompt,
temperature 0.1, row format) is imported unchanged from generate_answers_cached.py.

Resumable: a query is skipped when all four output files already hold it; a
query present in only some files is finished for the missing arms.

Run from the repository root:
    python rag/extensions/generate_length_arms.py --split dev \
        --transcript-cache rag/.cache/dev_transcripts.jsonl \
        --out-dir rag/results/phase5/answer_length
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_answers_cached as g  # noqa: E402
from generate_answers_resumable import generate_with_retry, read_existing, rewrite  # noqa: E402

SHORT_LINE = "Answer in at most 100 words."
ARMS = {  # name: (k, max_tokens, extra instruction)
    "k5": (5, None, None),
    "k2": (2, None, None),
    "k5_cap150": (5, 150, None),
    "k5_short": (5, None, SHORT_LINE),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--api-retries", type=int, default=6)
    args = ap.parse_args()

    cache = g.load_cache(args.transcript_cache)
    data_dir = g.ROOT / "data/rag_corpus"
    data = g.CourseData(data_dir)
    items = data.audio_items(args.split)
    missing = [it["query_id"] for it in items if it["query_id"] not in cache]
    if missing:
        sys.exit(f"Missing from cache: {len(missing)} queries, e.g. {missing[:3]}")
    if args.limit:
        items = items[:args.limit]

    done = {}
    for arm in ARMS:
        path = args.out_dir / arm / "answers.jsonl"
        rows, dropped = read_existing(path)
        for row in rows:
            if row.get("split") != args.split or row.get("config") != arm:
                sys.exit(f"{path} holds rows from another configuration; refusing to mix.")
        if path.exists():
            rewrite(path, rows)  # removes a truncated last line
        done[arm] = {row["query_id"] for row in rows}
        print(f"[resume] {arm}: {len(rows)} done, {dropped} truncated line(s) dropped", flush=True)

    encoder = g.LocalEncoder(g.ROOT / "models/enc_a", device=args.device)
    retriever = g.ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    builder = g.SimpleContextBuilder()
    generators = {None: g.ELMGeneratorWithUsage(), 150: g.ELMGeneratorWithUsage(max_tokens=150)}

    files = {}
    for arm in ARMS:
        (args.out_dir / arm).mkdir(parents=True, exist_ok=True)
        files[arm] = (args.out_dir / arm / "answers.jsonl").open("a", encoding="utf-8")
    names = list(ARMS)
    try:
        for i, item in enumerate(items):
            qid = item["query_id"]
            todo = [a for a in names[i % 4:] + names[:i % 4] if qid not in done[a]]  # rotated order
            if not todo:
                continue
            transcript = cache[qid]
            t0 = time.perf_counter()
            qvec = encoder.encode(transcript)
            embedding_ms = (time.perf_counter() - t0) * 1000.0
            for arm in todo:
                k, cap, line = ARMS[arm]
                ret = retriever.retrieve(qvec, k=k)
                built = builder.build(transcript, ret["record_ids"], data)
                prompt = built["prompt"] + (f"\n\n{line}" if line else "")
                gen = generate_with_retry(generators[cap], prompt, args.api_retries, qid)
                row = {
                    "query_id": qid, "split": args.split, "config": arm, "k": k,
                    "evidence": "dense", "max_tokens": cap, "instruction": line,
                    "order_in_query": todo.index(arm), "query_index": i,
                    "transcript": transcript, "answer": gen["answer"],
                    "retrieved_record_ids": ret["record_ids"],
                    "prompt_chars": len(prompt), "prompt_tokens": gen["prompt_tokens"],
                    "answer_chars": len(gen["answer"]), "answer_tokens": gen["answer_tokens"],
                    "finish_reason": gen["finish_reason"],
                    "timing": {"embedding_ms": embedding_ms, "retrieval_ms": ret["latency_ms"],
                               "generation_ms": gen["latency_ms"]},
                }
                files[arm].write(json.dumps(row, ensure_ascii=False) + "\n")
                files[arm].flush()
                print(qid, arm, gen["prompt_tokens"], gen["answer_tokens"], gen["finish_reason"],
                      round(gen["latency_ms"]), flush=True)
    finally:
        for f in files.values():
            f.close()

    short = [a for a in ARMS if len(read_existing(args.out_dir / a / "answers.jsonl")[0]) != len(items)]
    print(f"[done] incomplete arms: {short or 'none'}", flush=True)
    sys.exit(1 if short else 0)


if __name__ == "__main__":
    main()

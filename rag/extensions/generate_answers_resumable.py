"""Resumable wrapper around generate_answers_cached.py.

Generation is identical to generate_answers_cached.py: the same generator class,
system prompt, temperature, retrieval, context builder and output row format are
imported from that file. This wrapper only adds:
  1. resume: query IDs already in the output file are skipped, new rows are appended;
  2. retry with back-off when an ELM request raises.

Run from the repository root, e.g.
    python rag/extensions/generate_answers_resumable.py --split practice \
        --transcript-cache rag/.cache/practice_transcripts.jsonl \
        --config k5 --k 5 --output rag/results/practice_depth/k5/answers.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_answers_cached as g  # noqa: E402  (same directory)


def read_existing(path: Path):
    """Return the complete rows already written; drop a truncated final line."""
    rows, dropped = [], 0
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    dropped += 1
    return rows, dropped


def rewrite(path: Path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def generate_with_retry(generator, prompt, retries, qid):
    for attempt in range(retries):
        try:
            return generator.generate(prompt)
        except Exception as e:  # ELM/network error: wait and try again
            wait = min(30 * 2 ** attempt, 300)
            print(f"[retry] {qid}: attempt {attempt + 1}/{retries} failed: {e!r}; "
                  f"waiting {wait}s", flush=True)
            if attempt + 1 == retries:
                raise
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--evidence", choices=["dense", "gold_first", "gold_last"], default="dense")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--api-retries", type=int, default=6)
    args = ap.parse_args()

    cache = g.load_cache(args.transcript_cache)
    data_dir = g.ROOT / "data/rag_corpus"
    data = g.CourseData(data_dir)
    items = data.audio_items(args.split)
    qrels = data.load_qrels(args.split) if args.evidence != "dense" else {}

    missing = [it["query_id"] for it in items if it["query_id"] not in cache]
    if missing:
        sys.exit(f"Missing from cache: {len(missing)} queries, e.g. {missing[:3]}")
    if args.limit:
        items = items[:args.limit]

    done_rows, dropped = read_existing(args.output)
    for row in done_rows:
        same = (row.get("split") == args.split and row.get("config") == args.config
                and row.get("k") == args.k and row.get("evidence") == args.evidence)
        if not same:
            sys.exit(f"{args.output} holds rows from a different configuration "
                     f"({row.get('split')}, {row.get('config')}, k={row.get('k')}); refusing to mix.")
    done = {row["query_id"] for row in done_rows}
    if len(done) != len(done_rows):
        sys.exit(f"{args.output} contains duplicate query IDs; inspect it before resuming.")
    if args.output.exists():
        rewrite(args.output, done_rows)  # also removes a truncated last line
    todo = [it for it in items if it["query_id"] not in done]
    print(f"[resume] {len(done)} done, {len(todo)} to do, {dropped} truncated line(s) dropped",
          flush=True)
    if not todo:
        print("[done] nothing to generate", flush=True)
        return

    encoder = g.LocalEncoder(g.ROOT / "models/enc_a", device=args.device)
    retriever = g.ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    generator = g.ELMGeneratorWithUsage()
    builder = g.SimpleContextBuilder()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        for item in todo:
            qid = item["query_id"]
            transcript = cache[qid]

            t0 = time.perf_counter()
            qvec = encoder.encode(transcript)
            embedding_ms = (time.perf_counter() - t0) * 1000.0

            record_ids, ret = g.select_evidence(retriever, qvec, args.k,
                                                args.evidence, qrels.get(qid, {}))
            built = builder.build(transcript, record_ids, data)
            gen = generate_with_retry(generator, built["prompt"], args.api_retries, qid)

            row = {
                "query_id": qid,
                "split": args.split,
                "config": args.config,
                "k": args.k,
                "evidence": args.evidence,
                "transcript": transcript,
                "answer": gen["answer"],
                "retrieved_record_ids": record_ids,
                "prompt_chars": len(built["prompt"]),
                "prompt_tokens": gen["prompt_tokens"],
                "answer_chars": len(gen["answer"]),
                "answer_tokens": gen["answer_tokens"],
                "finish_reason": gen["finish_reason"],
                "timing": {
                    "embedding_ms": embedding_ms,
                    "retrieval_ms": ret["latency_ms"],
                    "generation_ms": gen["latency_ms"],
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(qid, args.config, gen["prompt_tokens"], gen["answer_tokens"],
                  round(gen["latency_ms"]), flush=True)

    final_rows, _ = read_existing(args.output)
    print(f"[done] {len(final_rows)}/{len(items)} rows in {args.output}", flush=True)
    if len(final_rows) != len(items):
        sys.exit(1)


if __name__ == "__main__":
    main()

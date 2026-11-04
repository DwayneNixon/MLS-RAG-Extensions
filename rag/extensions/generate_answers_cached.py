"""Generate answers from cached transcripts (no ASR), logging extra fields."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]      # the rag/ folder
sys.path.insert(0, str(ROOT))
import argparse
import json
import time
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.context_builder import SimpleContextBuilder

def load_cache(path: Path):
    cache = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["query_id"]] = row["transcript"]
    return cache

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--config", required=True, help="e.g. baseline, k10, k20")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.output.exists():
        ap.error("Results cannot be overwritten.")                               

    cache = load_cache(args.transcript_cache)
    
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)

    missing = [item["query_id"] for item in items if item["query_id"] not in cache]
    if missing:
        sys.exit(f"Missing from cache: {len(missing)} queries, e.g. {missing[:3]}")

    if args.limit:
        items = items[:args.limit]

    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    builder = SimpleContextBuilder()

    for item in items:
        qid = item["query_id"]
        transcript = cache[qid]

        t0 = time.perf_counter()
        qvec = encoder.encode(transcript)
        embedding_ms = (time.perf_counter() - t0) * 1000.0

        ret = retriever.retrieve(qvec, k=args.k)
        built = builder.build(transcript, ret["record_ids"], data)
        prompt_chars = len(built["prompt"])

        print(qid, len(ret["record_ids"]), "records,", prompt_chars, "chars,",
            round(embedding_ms, 1), "ms_embed,", round(ret["latency_ms"], 1), "ms retrieve")

if __name__ == "__main__":
    main()
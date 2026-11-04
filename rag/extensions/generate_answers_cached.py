from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]      # the rag/ folder
sys.path.insert(0, str(ROOT))
import argparse
import json

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
    
    from src.data import CourseData
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)

    missing = [item["query_id"] for item in items if item["query_id"] not in cache]
    if missing:
        sys.exit(f"Missing from cache: {len(missing)} queries, e.g. {missing[:3]}")

    print(f"All {len(items)} {args.split} queries found in the cache.")

if __name__ == "__main__":
    main()
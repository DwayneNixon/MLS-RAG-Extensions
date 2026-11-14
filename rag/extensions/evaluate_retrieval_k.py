"""Retrieval metrics for several k from cached transcripts (no ASR, no ELM)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import numpy as np
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from extensions.generate_answers_cached import load_cache

def precision_at_k(ranked_ids, relevant_ids, k):
    top_k = ranked_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / k

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Results cannot be overwritten.")

    ks = [2, 5, 10, 20, 100]
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    qrels = data.load_qrels(args.split)
    cache = load_cache(args.transcript_cache)

    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )

    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for item in items:
            qid = item["query_id"]
            qvec = encoder.encode(cache[qid])
            ranked = retriever.retrieve(qvec, k=max(ks))["record_ids"]
            rel = qrels[qid]

            row = {"query_id": qid, "MRR@10": mrr_at_k(ranked, rel, 10)}
            for k in ks:
                row[f"Recall@{k}"] = recall_at_k(ranked, rel, k)
                row[f"Precision@{k}"] = precision_at_k(ranked, rel, k)
                row[f"NDCG@{k}"] = ndcg_at_k(ranked, rel, k)
            rows.append(row)
            f.write(json.dumps(row) + "\n")
    print(f"{args.split}: {len(rows)} queries")
    for key in rows[0]:
        if key != "query_id":
            print(f"  {key:<14} {np.mean([r[key] for r in rows]):.4f}")

if __name__ == "__main__":
    main()
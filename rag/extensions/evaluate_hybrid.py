"""Evaluate hybrid reranking against the original dense retriever."""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from extensions.generate_answers_cached import load_cache
from extensions.hybrid_reranker import HybridReranker


def metrics(ranked_ids, qrels):
    return {
        "Recall@5": recall_at_k(ranked_ids, qrels, 5),
        "MRR@10": mrr_at_k(ranked_ids, qrels, 10),
        "NDCG@10": ndcg_at_k(ranked_ids, qrels, 10),
        "Recall@100": recall_at_k(ranked_ids, qrels, 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["practice", "dev"],
        default="practice"
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        parser.error("alpha must be between 0 and 1")

    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")

    if args.output.exists():
        parser.error("Output already exists; refusing to overwrite")

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    qrels = data.load_qrels(args.split)

    cache = load_cache(
        ROOT / f".cache/{args.split}_transcripts.jsonl"
    )

    items = data.audio_items(args.split)

    if args.limit is not None:
        items = items[:args.limit]

    encoder = LocalEncoder(
        ROOT / "models/enc_a",
        device=args.device
    )

    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json"
    )

    hybrid = HybridReranker(data, alpha=args.alpha)

    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("===== HYBRID RETRIEVAL EVALUATION =====")
    print("Split:", args.split)
    print("Queries:", len(items))
    print("Alpha:", args.alpha)
    print("Index build (ms):", round(hybrid.index_build_ms, 2))

    with args.output.open("x", encoding="utf-8") as f:
        for index, item in enumerate(items, start=1):
            qid = item["query_id"]
            transcript = cache[qid]

            start = time.perf_counter()
            qvec = encoder.encode(transcript)
            embedding_ms = (time.perf_counter() - start) * 1000

            dense = retriever.retrieve(qvec, k=100)

            result = hybrid.rerank(
                transcript, dense["record_ids"], top_k=5
            )

            original = dense["record_ids"]
            reranked = result["record_ids"]

            assert len(original) == 100
            assert len(reranked) == 100
            assert len(set(reranked)) == 100
            assert set(original) == set(reranked)

            baseline = metrics(original, qrels[qid])
            improved = metrics(reranked, qrels[qid])

            assert abs(
                baseline["Recall@100"]
                - improved["Recall@100"]
            ) < 1e-12

            row = {
                "query_id": qid,
                "split": args.split,
                "config": "hybrid",
                "alpha": args.alpha,
                "transcript": transcript,
                "dense_record_ids": original,
                "reranked_record_ids": reranked,
                "retrieved_record_ids": reranked[:5],
                "dense_metrics": baseline,
                "hybrid_metrics": improved,
                "timing": {
                    "embedding_ms": embedding_ms,
                    "retrieval_ms": dense["latency_ms"],
                    "rerank_ms": result["rerank_ms"],
                },
            }

            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            delta = (
                improved["Recall@5"] - baseline["Recall@5"]
            )

            print(
                f"{index:3d}/{len(items)} {qid} "
                f"R@5 {baseline['Recall@5']:.3f} "
                f"-> {improved['Recall@5']:.3f} "
                f"delta={delta:+.3f} "
                f"rerank={result['rerank_ms']:.2f}ms",
                flush=True,
            )

    print("\n===== SUMMARY =====")

    for metric in ["Recall@5", "MRR@10", "NDCG@10"]:
        before = mean(
            row["dense_metrics"][metric] for row in rows
        )
        after = mean(
            row["hybrid_metrics"][metric] for row in rows
        )

        print(
            f"{metric}: dense={before:.6f}, "
            f"hybrid={after:.6f}, "
            f"delta={after-before:+.6f}"
        )

    changes = [
        row["hybrid_metrics"]["Recall@5"]
        - row["dense_metrics"]["Recall@5"]
        for row in rows
    ]

    print("Queries improved:", sum(v > 1e-12 for v in changes))
    print("Queries unchanged:", sum(abs(v) <= 1e-12 for v in changes))
    print("Queries worsened:", sum(v < -1e-12 for v in changes))

    times = [row["timing"]["rerank_ms"] for row in rows]
    print("Mean rerank_ms:", round(mean(times), 3))
    print("Median rerank_ms:", round(median(times), 3))
    print("Results saved:", args.output)
    print("\nHYBRID EVALUATION: PASS")


if __name__ == "__main__":
    main()

"""Evaluate MiniLM reranking using previously saved dense top-100 results."""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from extensions.cross_encoder_reranker import MiniLMReranker


MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--cross-encoder--ms-marco-MiniLM-L6-v2"
    / "snapshots/233902d25c440f23af6f7d6e94d2946bac0bee0a"
)


def calculate_metrics(ids, relevant):
    return {
        "Recall@5": recall_at_k(ids, relevant, 5),
        "MRR@10": mrr_at_k(ids, relevant, 10),
        "NDCG@10": ndcg_at_k(ids, relevant, 10),
        "Recall@100": recall_at_k(ids, relevant, 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["practice", "dev"],
        default="practice"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.limit < 1 or args.batch_size < 1:
        parser.error("Limit and batch size must be positive")

    if args.output.exists():
        parser.error("Output exists: refusing to overwrite")

    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 2080 Ti" not in gpu_name:
        parser.error(f"Wrong GPU: {gpu_name}")

    source_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(source_rows) < args.limit:
        parser.error("Not enough input queries")

    source_rows = source_rows[:args.limit]

    assert all(
        row["split"] == args.split
        for row in source_rows
    )

    assert len({
        row["query_id"] for row in source_rows
    }) == len(source_rows)

    data = CourseData(ROOT / "data/rag_corpus")
    qrels = data.load_qrels(args.split)

    reranker = MiniLMReranker(
        model_path=MODEL_PATH,
        device="cuda",
        batch_size=args.batch_size,
        max_length=512,
    )

    # Warm up with real document pairs before collecting timings.
    first = source_rows[0]
    warmup_ids = first["dense_record_ids"][:args.batch_size]
    warmup_pairs = [
        (first["transcript"], data.record_text(rid))
        for rid in warmup_ids
    ]

    reranker.model.predict(
        warmup_pairs,
        batch_size=args.batch_size,
        show_progress_bar=False,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("===== MINILM RETRIEVAL EVALUATION =====")
    print("GPU:", gpu_name)
    print("Split:", args.split)
    print("Queries:", len(source_rows))
    print("Batch size:", args.batch_size)
    print("Model revision:", MODEL_PATH.name)
    print(
        "Model load (ms):",
        round(reranker.model_load_ms, 2)
    )

    results = []

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as f:
        for i, source in enumerate(source_rows, start=1):
            qid = source["query_id"]
            transcript = source["transcript"]
            dense_ids = source["dense_record_ids"]
            relevant = qrels[qid]

            assert len(dense_ids) == 100
            assert len(set(dense_ids)) == 100

            result = reranker.rerank(
                query=transcript,
                candidate_ids=dense_ids,
                data=data,
                top_k=5,
            )

            minilm_ids = result["record_ids"]

            assert len(minilm_ids) == 100
            assert set(minilm_ids) == set(dense_ids)

            dense_metrics = calculate_metrics(
                dense_ids, relevant
            )

            minilm_metrics = calculate_metrics(
                minilm_ids, relevant
            )

            assert abs(
                dense_metrics["Recall@100"]
                - minilm_metrics["Recall@100"]
            ) < 1e-12

            assert abs(
                dense_metrics["Recall@5"]
                - source["dense_metrics"]["Recall@5"]
            ) < 1e-12

            row = {
                "query_id": qid,
                "split": args.split,
                "config": "minilm_top100_to5",
                "model_revision": MODEL_PATH.name,
                "batch_size": args.batch_size,
                "max_length": 512,
                "transcript": transcript,
                "dense_record_ids": dense_ids,
                "reranked_record_ids": minilm_ids,
                "retrieved_record_ids": minilm_ids[:5],
                "reranked_scores": result["scores"],
                "dense_metrics": dense_metrics,
                "minilm_metrics": minilm_metrics,
                "timing": {
                    "rerank_ms": result["rerank_ms"],
                },
            }

            results.append(row)
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            f.flush()

            before = dense_metrics["Recall@5"]
            after = minilm_metrics["Recall@5"]

            print(
                f"{i:3d}/{len(source_rows)} {qid} "
                f"R@5 {before:.3f} -> {after:.3f} "
                f"delta={after-before:+.3f} "
                f"rerank={result['rerank_ms']:.2f}ms",
                flush=True
            )

    print("\n===== SUMMARY =====")

    for metric in ["Recall@5", "MRR@10", "NDCG@10"]:
        before = mean(
            row["dense_metrics"][metric]
            for row in results
        )

        after = mean(
            row["minilm_metrics"][metric]
            for row in results
        )

        print(
            f"{metric}: "
            f"dense={before:.6f}, "
            f"minilm={after:.6f}, "
            f"delta={after-before:+.6f}"
        )

    changes = [
        row["minilm_metrics"]["Recall@5"]
        - row["dense_metrics"]["Recall@5"]
        for row in results
    ]

    print(
        "Queries improved:",
        sum(x > 1e-12 for x in changes)
    )
    print(
        "Queries unchanged:",
        sum(abs(x) <= 1e-12 for x in changes)
    )
    print(
        "Queries worsened:",
        sum(x < -1e-12 for x in changes)
    )

    timings = [
        row["timing"]["rerank_ms"]
        for row in results
    ]

    print("Mean rerank_ms:", round(mean(timings), 3))
    print("Median rerank_ms:", round(median(timings), 3))

    print(
        "Peak PyTorch allocated GPU memory (GiB):",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3
        )
    )

    print("Saved:", args.output)
    print("\nMINILM EVALUATION: PASS")


if __name__ == "__main__":
    main()

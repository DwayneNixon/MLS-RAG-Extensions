"""Sweep hybrid fusion weights using saved Practice dense candidates."""

import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from extensions.hybrid_reranker import HybridReranker

INPUT = ROOT / "results/hybrid_practice_alpha050.jsonl"
OUTPUT = ROOT / "results/hybrid_alpha_sweep_practice.csv"

ALPHAS = [0.0, 0.25, 0.5, 0.65, 0.75,
          0.85, 0.9, 0.95, 0.98, 1.0]

rows = [
    json.loads(line)
    for line in INPUT.read_text().splitlines()
    if line.strip()
]

assert len(rows) == 350

data = CourseData(ROOT / "data/rag_corpus")
qrels = data.load_qrels("practice")
reranker = HybridReranker(data, alpha=0.5)

results = {
    alpha: {
        "Recall@5": [],
        "MRR@10": [],
        "NDCG@10": [],
        "improved": 0,
        "unchanged": 0,
        "worsened": 0,
    }
    for alpha in ALPHAS
}

for row in rows:
    qid = row["query_id"]
    dense_ids = row["dense_record_ids"]
    relevant = qrels[qid]

    assert len(dense_ids) == 100
    assert len(set(dense_ids)) == 100

    # Calculate BM25 ranks once for this question.
    hybrid = reranker.rerank(
        row["transcript"], dense_ids, top_k=5
    )

    bm25_ranks = dict(zip(
        hybrid["record_ids"],
        hybrid["bm25_ranks"],
    ))

    baseline_recall = recall_at_k(dense_ids, relevant, 5)

    for alpha in ALPHAS:
        order = sorted(
            range(len(dense_ids)),
            key=lambda i: (
                -(
                    alpha / (60 + i + 1)
                    + (1 - alpha)
                    / (60 + bm25_ranks[dense_ids[i]])
                ),
                i,
            ),
        )

        ranked = [dense_ids[i] for i in order]

        # Verify both endpoints and our original result.
        if alpha == 0.5:
            assert ranked == row["reranked_record_ids"]

        if alpha == 1.0:
            assert ranked == dense_ids

        scores = results[alpha]

        recall = recall_at_k(ranked, relevant, 5)
        scores["Recall@5"].append(recall)
        scores["MRR@10"].append(
            mrr_at_k(ranked, relevant, 10)
        )
        scores["NDCG@10"].append(
            ndcg_at_k(ranked, relevant, 10)
        )

        delta = recall - baseline_recall

        if delta > 1e-12:
            scores["improved"] += 1
        elif delta < -1e-12:
            scores["worsened"] += 1
        else:
            scores["unchanged"] += 1

summary = []

for alpha in ALPHAS:
    scores = results[alpha]

    summary.append({
        "alpha": alpha,
        "Recall@5": mean(scores["Recall@5"]),
        "MRR@10": mean(scores["MRR@10"]),
        "NDCG@10": mean(scores["NDCG@10"]),
        "improved": scores["improved"],
        "unchanged": scores["unchanged"],
        "worsened": scores["worsened"],
    })

assert not OUTPUT.exists(), "Refusing to overwrite results"

with OUTPUT.open("x", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("===== HYBRID ALPHA SWEEP: PRACTICE =====")
print("Queries:", len(rows))
print("BM25 index build (ms):",
      round(reranker.index_build_ms, 2))
print()
print(
    f"{'Alpha':>6} {'R@5':>9} {'MRR@10':>9} "
    f"{'NDCG@10':>9} {'Better':>7} {'Same':>6} {'Worse':>7}"
)

for row in summary:
    print(
        f"{row['alpha']:6.2f} "
        f"{row['Recall@5']:9.6f} "
        f"{row['MRR@10']:9.6f} "
        f"{row['NDCG@10']:9.6f} "
        f"{row['improved']:7d} "
        f"{row['unchanged']:6d} "
        f"{row['worsened']:7d}"
    )

print("\nOutput:", OUTPUT)
print("ALPHA SWEEP: PASS")

#!/usr/bin/env python3
"""Matched-node Hybrid vs MiniLM reranking benchmark.

Saved dense top-100 inputs, 123 common development queries.
No ELM calls, answer generation, or dense retrieval.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--cross-encoder--ms-marco-MiniLM-L6-v2"
    / "snapshots/233902d25c440f23af6f7d6e94d2946bac0bee0a"
)


def load(path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {r["query_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate query IDs: {path}")
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_ms": st.mean(ordered),
        "median_ms": st.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def prepare():
    base = ROOT / "results"
    unified_path = base / "reranking_unified_dev123.json"
    hybrid_path = base / "hybrid_dev_alpha095.jsonl"
    minilm_path = base / "minilm_dev_full125.jsonl"

    unified = json.loads(unified_path.read_text())
    qids = unified["query_ids"]

    if len(qids) != 123 or len(set(qids)) != 123:
        raise ValueError("Expected 123 unique common queries")

    hybrid = load(hybrid_path)
    minilm = load(minilm_path)
    data = CourseData(ROOT / "data/rag_corpus")

    items = []

    for q in qids:
        h = hybrid[q]
        m = minilm[q]

        assert h["transcript"] == m["transcript"]
        assert h["dense_record_ids"] == m["dense_record_ids"]

        ids = h["dense_record_ids"]

        assert len(ids) == len(set(ids)) == 100
        assert all(rid in data.records for rid in ids)
        assert abs(h["alpha"] - 0.95) < 1e-12

        items.append({
            "query_id": q,
            "transcript": h["transcript"],
            "candidate_ids": ids,
            "expected_hybrid": h["reranked_record_ids"][:5],
            "expected_minilm": m["reranked_record_ids"][:5],
        })

    return items, data, {
        "unified": sha256(unified_path),
        "hybrid": sha256(hybrid_path),
        "minilm": sha256(minilm_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.repeats < 1 or args.warmup < 1:
        parser.error("Repeats and warmup must be positive")

    if not args.validate_only:
        if args.output is None:
            parser.error("--output is required for benchmarking")
        if args.output.exists():
            parser.error("Output already exists; refusing to overwrite")

    items, data, input_hashes = prepare()

    print("===== MATCHED BENCHMARK INPUTS =====")
    print("Node:", socket.gethostname())
    print("Queries:", len(items))
    print("Candidate documents per query:", 100)
    print("Identical transcripts and candidate rankings: PASS")
    print("BM25 alpha:", 0.95)
    print("MiniLM model available:", MODEL_PATH.is_dir())

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)

    if args.validate_only:
        print("\nPASS: Validation-only mode")
        print("No model loaded, GPU used, or results written")
        return

    import torch
    from extensions.hybrid_reranker import HybridReranker
    from extensions.cross_encoder_reranker import MiniLMReranker

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run benchmark inside a Slurm allocation")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    gpu = torch.cuda.get_device_name(0)
    if "RTX 2080 Ti" not in gpu:
        raise RuntimeError(f"Unexpected GPU: {gpu}")

    torch.set_num_threads(2)

    print("\n===== MODEL SETUP =====", flush=True)

    hybrid = HybridReranker(data, alpha=0.95)
    minilm = MiniLMReranker(
        model_path=MODEL_PATH,
        device="cuda",
        batch_size=16,
        max_length=512,
    )

    def run(method, item):
        if method == "Hybrid":
            result = hybrid.rerank(
                item["transcript"],
                item["candidate_ids"],
                top_k=5,
            )
        else:
            result = minilm.rerank(
                query=item["transcript"],
                candidate_ids=item["candidate_ids"],
                data=data,
                top_k=5,
            )

        assert result["candidate_count"] == 100
        assert len(result["record_ids"]) == 100
        assert set(result["record_ids"]) == set(item["candidate_ids"])

        return result

    print("GPU:", gpu)
    print("Hybrid index build (ms):", hybrid.index_build_ms)
    print("MiniLM model load (ms):", minilm.model_load_ms)

    print("\n===== WARM-UP =====", flush=True)

    for item in items[:args.warmup]:
        run("Hybrid", item)
        run("MiniLM", item)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    measured = {
        item["query_id"]: {
            "Hybrid": [],
            "MiniLM": [],
        }
        for item in items
    }

    mismatches = {"Hybrid": set(), "MiniLM": set()}

    print("\n===== MEASURED RUNS =====", flush=True)

    # Alternate method order to reduce systematic timing drift.
    for repeat in range(args.repeats):
        print(
            f"Pass {repeat + 1}/{args.repeats}",
            flush=True,
        )

        for index, item in enumerate(items):
            order = (
                ("Hybrid", "MiniLM")
                if (repeat + index) % 2 == 0
                else ("MiniLM", "Hybrid")
            )

            for method in order:
                result = run(method, item)

                measured[item["query_id"]][method].append(
                    result["rerank_ms"]
                )

                expected = item[
                    "expected_hybrid"
                    if method == "Hybrid"
                    else "expected_minilm"
                ]

                if result["record_ids"][:5] != expected:
                    mismatches[method].add(item["query_id"])

    print("\n===== TIMING SUMMARY =====")

    summary = {}

    for method in ("Hybrid", "MiniLM"):
        values = [
            t
            for item in items
            for t in measured[item["query_id"]][method]
        ]

        summary[method] = stats(values)

        print(f"\n{method}")
        for key, value in summary[method].items():
            print(f"  {key}: {value}")

        print(
            "  Top-5 mismatches against saved results:",
            len(mismatches[method]),
        )

    result = {
        "experiment": "matched_node_reranking_dev123",
        "hardware": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "gpu": gpu,
            "gpu_total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory
            ),
            "peak_torch_allocated_gpu_bytes": (
                torch.cuda.max_memory_allocated()
            ),
        },
        "configuration": {
            "queries": len(items),
            "candidates_per_query": 100,
            "warmup_queries": args.warmup,
            "measured_passes": args.repeats,
            "hybrid_alpha": 0.95,
            "minilm_model_revision": MODEL_PATH.name,
            "minilm_batch_size": 16,
            "minilm_max_length": 512,
            "python": platform.python_version(),
        },
        "input_sha256": input_hashes,
        "setup_ms": {
            "hybrid_index_build": hybrid.index_build_ms,
            "minilm_model_load": minilm.model_load_ms,
        },
        "summary": summary,
        "ranking_mismatch_queries": {
            method: sorted(values)
            for method, values in mismatches.items()
        },
        "measurements": measured,
        "notes": [
            "Both methods ran on the same compute node.",
            "Hybrid used CPU; MiniLM used RTX 2080 Ti GPU.",
            "Each measurement reranks exactly 100 candidates.",
            "Timing excludes embedding, dense retrieval, generation and judging.",
            "Setup times are reported separately.",
            "Repeated timings are not independent query samples.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print("\nPASS: Matched-node benchmark completed")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

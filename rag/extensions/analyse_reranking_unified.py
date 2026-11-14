#!/usr/bin/env python3
"""Unified 123-query Hybrid/MiniLM analysis against k5 judge repeat 1.

Reads saved results only. No model inference or ELM API calls.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF = "f54c766616cf247306613d01a565e7c882a72413"
DIMENSIONS = ("correctness", "grounding", "completeness")
DEPTHS = ("k2", "k5", "k10", "k20", "k5b")


def read_jsonl(text):
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def load_local(path):
    return read_jsonl((ROOT / path).read_text(encoding="utf-8"))


def load_git(ref, path, repo):
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        text=True,
    )
    return read_jsonl(raw)


def indexed(rows):
    result = {r["query_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate query IDs found")
    return result


def describe(values):
    n = len(values)
    if n < 2:
        raise ValueError("At least two observations required")

    avg = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)

    return {
        "n": n,
        "mean": avg,
        "se": se,
        "ci95": [avg - 1.96 * se, avg + 1.96 * se],
        "better": sum(v > 1e-9 for v in values),
        "tied": sum(abs(v) <= 1e-9 for v in values),
        "worse": sum(v < -1e-9 for v in values),
    }


def show(label, result):
    print(
        f"{label:25s} n={result['n']:3d} "
        f"mean={result['mean']:+.6f} "
        f"SE={result['se']:.6f} "
        f"CI=[{result['ci95'][0]:+.6f}, "
        f"{result['ci95'][1]:+.6f}]"
    )


def zero(x):
    return abs(x) < 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teammate-ref", default=DEFAULT_REF)
    parser.add_argument(
        "--teammate-repo", type=Path,
        default=Path.home() / "MLS-RAG-Extensions"
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path.home()
        / "edin-mls-26-autumn/rag/data/rag_corpus/qrels_dev.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "rag/results/reranking_unified_dev123.json",
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists; refusing to overwrite")

    # Teammate's judged k-depth results.
    methods = {}
    for k in DEPTHS:
        path = (
            f"rag/results/phase3/{k}/judge/"
            "answer_judge_per_query.jsonl"
        )
        methods[k] = indexed(load_git(
            args.teammate_ref, path, args.teammate_repo
        ))

    # Our judged reranking results.
    methods["Hybrid"] = indexed(load_local(
        "rag/results/hybrid_dev_judge_full125/"
        "answer_judge_per_query.jsonl"
    ))
    methods["MiniLM"] = indexed(load_local(
        "rag/results/minilm_dev_judge_full125/"
        "answer_judge_per_query.jsonl"
    ))

    retrieval = indexed(load_local(
        "rag/results/minilm_dev_full125.jsonl"
    ))

    # Exact intersection across all configurations.
    common = sorted(set.intersection(
        *(set(x) for x in [*methods.values(), retrieval])
    ))
    assert len(common) == 123, (
        f"Expected 123 common queries, found {len(common)}"
    )

    # Individual k5 judge repeats, in original cache order.
    repeats = defaultdict(list)
    for row in load_git(
        args.teammate_ref,
        "rag/results/phase3/k5/judge/judge_cache.jsonl",
        args.teammate_repo,
    ):
        repeats[row["query_id"]].append(
            sum(row["result"][k] for k in DIMENSIONS) / 12
        )

    baseline = {}
    for q in common:
        assert len(repeats[q]) == 3
        assert math.isclose(
            statistics.mean(repeats[q]),
            methods["k5"][q]["normalised_answer_score"],
            abs_tol=1e-8,
        )
        baseline[q] = repeats[q][0]

    scores = {"k5": baseline}
    for name in ("k5b", "Hybrid", "MiniLM"):
        scores[name] = {
            q: methods[name][q]["normalised_answer_score"]
            for q in common
        }

    def differences(name, ids):
        return [scores[name][q] - baseline[q] for q in ids]

    print("===== UNIFIED ANALYSIS =====")
    print("Common queries:", len(common))
    print("Baseline: k5 judge repeat 1")
    print("Teammate commit:", args.teammate_ref)
    print("Baseline mean:", round(statistics.mean(baseline.values()), 6))

    overall = {}
    for name in ("k5b", "Hybrid", "MiniLM"):
        overall[name] = describe(differences(name, common))
        show(name + " - k5", overall[name])

    # MiniLM Recall@5 changes.
    recall_groups = {
        "Decreased": [],
        "Unchanged": [],
        "Increased": [],
    }

    for q in common:
        row = retrieval[q]
        delta = (
            row["minilm_metrics"]["Recall@5"]
            - row["dense_metrics"]["Recall@5"]
        )
        label = (
            "Decreased" if delta < -1e-9
            else "Increased" if delta > 1e-9
            else "Unchanged"
        )
        recall_groups[label].append(q)

    print("\n===== MINILM RECALL SUBGROUPS =====")
    subgroup_stats = {}
    for label, ids in recall_groups.items():
        subgroup_stats[label] = describe(differences("MiniLM", ids))
        show(label, subgroup_stats[label])

    assert [len(recall_groups[k]) for k in (
        "Decreased", "Unchanged", "Increased"
    )] == [23, 86, 14]

    # Zero-score counts and transitions.
    print("\n===== ZERO-SCORE ANALYSIS =====")
    zero_counts = {
        name: sum(zero(scores[name][q]) for q in common)
        for name in scores
    }
    for name, count in zero_counts.items():
        print(f"{name:10s}: {count}/123")

    persistent_zero = sum(
        all(zero(scores[name][q]) for name in scores)
        for q in common
    )
    print("Zero in all four:", persistent_zero)

    zero_transitions = {}
    for name in ("k5b", "Hybrid", "MiniLM"):
        recovered = sum(
            zero(baseline[q]) and not zero(scores[name][q])
            for q in common
        )
        dropped = sum(
            not zero(baseline[q]) and zero(scores[name][q])
            for q in common
        )
        zero_transitions[name] = {
            "zero_to_nonzero": recovered,
            "nonzero_to_zero": dropped,
        }
        print(f"{name}: recovered={recovered}, dropped={dropped}")

    assert persistent_zero == 34

    # Relevance labels.
    qrels = defaultdict(set)
    with args.qrels.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if float(row["score"]) > 0:
                qrels[row["query_id"]].add(row["record_id"])

    position_groups = {
        "Lost top-2": [],
        "Gained top-2": [],
        "Relevant in both": [],
        "Neither relevant": [],
    }
    top1_transitions = defaultdict(int)

    for q in common:
        row = retrieval[q]
        relevant = qrels[q]
        assert relevant, f"Missing qrels for {q}"

        dense_ids = row["dense_record_ids"]
        mini_ids = row["reranked_record_ids"]

        # Validate saved Recall@5 against the relevance labels.
        for ids, key in (
            (dense_ids, "dense_metrics"),
            (mini_ids, "minilm_metrics"),
        ):
            calculated = sum(
                rid in relevant for rid in ids[:5]
            ) / len(relevant)
            assert math.isclose(
                calculated, row[key]["Recall@5"], abs_tol=1e-6
            ), f"Recall mismatch: {q}"

        before2 = any(x in relevant for x in dense_ids[:2])
        after2 = any(x in relevant for x in mini_ids[:2])

        if before2 and not after2:
            label = "Lost top-2"
        elif not before2 and after2:
            label = "Gained top-2"
        elif before2 and after2:
            label = "Relevant in both"
        else:
            label = "Neither relevant"

        position_groups[label].append(q)

        top1_transitions[
            f"{dense_ids[0] in relevant}->{mini_ids[0] in relevant}"
        ] += 1

    print("\n===== TOP-2 RELEVANCE ANALYSIS =====")
    position_stats = {}
    for label, ids in position_groups.items():
        position_stats[label] = describe(differences("MiniLM", ids))
        show(label, position_stats[label])

    assert [len(position_groups[k]) for k in (
        "Lost top-2", "Gained top-2",
        "Relevant in both", "Neither relevant"
    )] == [15, 10, 41, 57]

    report = {
        "description": "Unified 123-query reranking analysis",
        "baseline": "k5 first cached judge repeat",
        "teammate_ref": args.teammate_ref,
        "query_count": len(common),
        "query_ids": common,
        "overall": overall,
        "minilm_recall_subgroups": subgroup_stats,
        "zero_counts": zero_counts,
        "persistent_zero_count": persistent_zero,
        "zero_transitions": zero_transitions,
        "top1_relevance_transitions": dict(top1_transitions),
        "top2_relevance_subgroups": position_stats,
        "limitations": [
            "One saved generation/judge observation per candidate.",
            "Paired CIs use 1.96 * SE and do not capture model stochasticity.",
            "Subgroups are exploratory and do not establish causality.",
            "Hybrid reuses previously judged k5b answers for some queries.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print("\nPASS: Unified analysis completed")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

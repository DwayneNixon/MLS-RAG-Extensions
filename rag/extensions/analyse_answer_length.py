"""Analysis for ANALYSIS_PLAN_answer_length.md. Standard library only.

    python rag/extensions/analyse_answer_length.py rag/results/phase5/answer_length
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "phase5" / "answer_length"
ARMS = ("k5", "k2", "k5_cap150", "k5_short")
JUDGED = ("k5_short", "k5_cap150")
SUBS = ("correctness", "grounding", "completeness")


def jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def se(x):
    return stdev(x) / math.sqrt(len(x))


def judged(p):
    """query_id -> dict of the three sub-scores (0-4) and the normalised score."""
    out = {}
    for r in jsonl(p):
        s = {c: float(r[c]) for c in SUBS}
        s["score"] = float(r["normalised_answer_score"])
        out[r["query_id"]] = s
    return out


def summary(label, d):
    m, s = mean(d), se(d)
    return f"{label}: n={len(d)}  mean {m:+.3f}  SE {s:.3f}  95% [{m - 1.96 * s:+.3f}, {m + 1.96 * s:+.3f}]"


# Saved references, defined exactly as in report/report_numbers.py
ref = {"k2": judged(RES / "phase3/k2/judge/answer_judge_per_query.jsonl")}
k5 = {}
for r in jsonl(RES / "phase3/k5/judge/judge_cache.jsonl"):
    if r["query_id"] not in k5:  # judge repeat 1
        x = r["result"]
        k5[r["query_id"]] = {c: float(x[c]) for c in SUBS}
        k5[r["query_id"]]["score"] = sum(x[c] for c in SUBS) / 12
ref["k5"] = k5
shared = set(ref["k2"]) & set(ref["k5"])
for c in ("k10", "k20", "k5b"):
    shared &= {r["query_id"] for r in jsonl(RES / f"phase3/{c}/judge/answer_judge_per_query.jsonl")}
for f in ("hybrid_dev_judge_full125", "minilm_dev_judge_full125"):
    shared &= {r["query_id"] for r in jsonl(RES / f / "answer_judge_per_query.jsonl")}
print(f"Development queries paired in every report configuration: {len(shared)}")

print("\n== 1. Latency and length, same session (first query of the run dropped as warm-up)")
rows = {a: jsonl(OUT / a / "answers.jsonl") for a in ARMS}
first = min(r["query_index"] for a in ARMS for r in rows[a])
rows = {a: [r for r in rows[a] if r["query_index"] != first] for a in ARMS}
med = {}
for a in ARMS:
    gm = [r["timing"]["generation_ms"] for r in rows[a]]
    med[a] = median(gm)
    print(f"{a:10s} n={len(gm)}  median answer tokens {median(r['answer_tokens'] for r in rows[a]):.0f}"
          f"  median prompt tokens {median(r['prompt_tokens'] for r in rows[a]):.0f}"
          f"  median generation ms {med[a]:.0f}"
          f"  finish_reason {dict(Counter(r['finish_reason'] for r in rows[a]))}")
by = {a: {r["query_id"]: r["timing"]["generation_ms"] for r in rows[a]} for a in ARMS}
for a in ("k5", "k5_cap150", "k5_short"):
    qs = sorted(set(by[a]) & set(by["k2"]))
    ratio = median(by[a][q] / by["k2"][q] for q in qs)
    verdict = "within 10% of k2" if abs(med[a] / med["k2"] - 1) <= 0.10 else "not within 10% of k2"
    print(f"{a} vs k2 re-run: ratio of medians {med[a] / med['k2']:.2f} ({verdict}); "
          f"median paired ratio {ratio:.2f} (n={len(qs)})")

print("\n== 2-3. Quality against saved references (Development, judge repeat 1 for k5)")
for a in JUDGED:
    p = OUT / a / "judge" / "answer_judge_per_query.jsonl"
    if not p.exists():
        print(f"{a}: not judged yet")
        continue
    new = judged(p)
    for refname in ("k5", "k2"):
        qs = sorted(shared & set(new))
        print(summary(f"{a} minus saved {refname}, score", [new[q]["score"] - ref[refname][q]["score"] for q in qs]))
        for c in SUBS:
            print("    " + summary(f"{c} (0-4)", [new[q][c] - ref[refname][q][c] for q in qs]))
    excl = OUT / a / "judge" / "excluded.jsonl"
    n_ex = sum(1 for l in excl.open() if l.strip()) if excl.exists() else 0
    print(f"  judge exclusions for {a}: {n_ex}")

print("\n== 4. Truncation in k5_cap150")
p = OUT / "k5_cap150" / "judge" / "answer_judge_per_query.jsonl"
if p.exists():
    new = judged(p)
    fin = {r["query_id"]: r["finish_reason"] for r in jsonl(OUT / "k5_cap150" / "answers.jsonl")}
    for lab, keep in (("truncated (finish_reason=length)", lambda f: f == "length"),
                      ("not truncated", lambda f: f != "length")):
        qs = sorted(q for q in shared & set(new) if keep(fin.get(q)))
        if len(qs) > 1:
            print(summary(f"k5_cap150 minus saved k5, {lab}", [new[q]["score"] - ref["k5"][q]["score"] for q in qs]))
        else:
            print(f"k5_cap150, {lab}: n={len(qs)}")

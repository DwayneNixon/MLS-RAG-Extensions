"""Analysis fixed in ANALYSIS_PLAN_practice_depth.md. Run from the repository root.

Reads:
  rag/results/practice_depth/{k5,k2,k20}/judge/answer_judge_per_query.jsonl
  rag/results/phase3/{k5,k2,k20}/judge/...   (Development, already reported)
Prints the primary and secondary results and writes analysis_summary.json.
"""
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
PRACTICE = ROOT / "results" / "practice_depth"
DEV = ROOT / "results" / "phase3"
MARGIN = 0.03          # non-inferiority margin, fixed in advance
Z90, Z95 = 1.645, 1.960


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_query(folder):
    path = folder / "judge" / "answer_judge_per_query.jsonl"
    if not path.exists():
        return None
    return {r["query_id"]: r["normalised_answer_score"] for r in load_jsonl(path)}


def dev_k5_first_repeat():
    """Development k=5 was judged three times; the reference is judge repeat 1."""
    first = {}
    for r in load_jsonl(DEV / "k5" / "judge" / "judge_cache.jsonl"):
        x = r["result"]
        first.setdefault(r["query_id"],
                         (x["correctness"] + x["grounding"] + x["completeness"]) / 12)
    return first


def paired(a, b):
    """Differences a - b on the queries scored in both."""
    return [a[q] - b[q] for q in sorted(set(a) & set(b))]


def summarise(d):
    m, se = mean(d), stdev(d) / math.sqrt(len(d))
    return {"n": len(d), "mean": m, "se": se,
            "ci90": [m - Z90 * se, m + Z90 * se], "ci95": [m - Z95 * se, m + Z95 * se]}


def show(label, s):
    print(f"{label}: n={s['n']}  mean {s['mean']:+.4f}  SE {s['se']:.4f}  "
          f"90% CI [{s['ci90'][0]:+.4f}, {s['ci90'][1]:+.4f}]  "
          f"95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]")


def verdict(s):
    lo, hi = s["ci90"]
    if lo > -MARGIN:
        return (f"NON-INFERIORITY ESTABLISHED: the lower 90% bound ({lo:+.4f}) is above "
                f"-{MARGIN}. k=2 loses less than {MARGIN} relative to k=5.")
    if hi < -MARGIN:
        return (f"INFERIOR: the upper 90% bound ({hi:+.4f}) is below -{MARGIN}. "
                f"k=2 loses more than {MARGIN}; the k=2 recommendation is withdrawn.")
    if hi < 0:
        return (f"k=2 IS WORSE: the 90% interval lies below zero, with a loss between "
                f"{-hi:.4f} and {-lo:.4f}; the recommendation must be qualified by that cost.")
    return (f"CONSISTENT WITH NON-INFERIORITY BUT NOT ESTABLISHED: the interval includes "
            f"zero but its lower bound ({lo:+.4f}) is not above -{MARGIN}. "
            f"The supported claim is that k=2 loses less than {-lo:.4f}.")


def main():
    p = {arm: per_query(PRACTICE / arm) for arm in ("k5", "k2", "k20")}
    if p["k5"] is None or p["k2"] is None:
        sys.exit("Practice k5 and k2 judgements are required for the primary analysis.")
    out = {"margin": MARGIN}

    for arm in ("k5", "k2", "k20"):
        ex = PRACTICE / arm / "judge" / "excluded.jsonl"
        ids = [r["query_id"] for r in load_jsonl(ex)] if ex.exists() else []
        n = len(p[arm]) if p[arm] else 0
        print(f"Practice {arm}: {n} judged, excluded {ids}")
        out[f"practice_{arm}_excluded"] = ids

    print("\n=== PRIMARY: Practice, k=2 minus k=5 (non-inferiority, margin 0.03) ===")
    s = summarise(paired(p["k2"], p["k5"]))
    show("Practice k2 - k5", s)
    print(verdict(s))
    out["primary_practice_k2_minus_k5"] = dict(s, verdict=verdict(s))

    dev = {"k5": dev_k5_first_repeat(), "k2": per_query(DEV / "k2"), "k20": per_query(DEV / "k20")}
    # Development comparisons use the same 123 queries as the report: those scored
    # in every depth configuration.
    shared = set(dev["k5"])
    for arm in ("k2", "k5b", "k10", "k20"):
        shared &= set(per_query(DEV / arm))
    dev = {arm: {q: v for q, v in scores.items() if q in shared} for arm, scores in dev.items()}
    print(f"\nDevelopment queries scored in every depth configuration: {len(shared)}")

    print("\n=== SECONDARY 1: Practice and Development pooled, k=2 minus k=5 ===")
    d_dev = paired(dev["k2"], dev["k5"])
    show("Development k2 - k5 (already reported)", summarise(d_dev))
    s = summarise(paired(p["k2"], p["k5"]) + d_dev)
    show("Pooled k2 - k5", s)
    print(verdict(s))
    out["secondary_pooled_k2_minus_k5"] = dict(s, verdict=verdict(s))

    print("\n=== SECONDARY 2: k=20 minus k=5 (two-sided estimate) ===")
    if p["k20"] is None:
        print("Practice k20 not available.")
    else:
        s = summarise(paired(p["k20"], p["k5"]))
        show("Practice k20 - k5", s)
        out["secondary_practice_k20_minus_k5"] = s
        d_dev20 = paired(dev["k20"], dev["k5"])
        show("Development k20 - k5 (already reported)", summarise(d_dev20))
        s = summarise(paired(p["k20"], p["k5"]) + d_dev20)
        show("Pooled k20 - k5", s)
        out["secondary_pooled_k20_minus_k5"] = s

    print("\n=== DESCRIPTIVE ===")
    for arm in ("k5", "k2", "k20"):
        if p[arm]:
            vals = list(p[arm].values())
            print(f"Practice {arm}: mean score {mean(vals):.4f}, "
                  f"share scoring exactly zero {sum(v == 0 for v in vals) / len(vals):.3f}")
    with open(PRACTICE / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

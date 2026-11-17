"""Recompute every number quoted in the report from the saved result files.

Run from anywhere:
    python report/report_numbers.py | tee report/report_numbers.txt

Each line shows: the value recomputed from the files, the value the report
quotes, OK or DIFF, and the file it came from. A file that cannot be found is
listed as MISSING at the end: that is a number a reader could not check.

Only the standard library is used. Nothing is generated or re-run: the script
reads saved answers, judge scores, retrieval metrics and benchmark logs.
"""
import csv
import glob
import json
import math
import random
import re
import sys
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
RAG = ROOT / "rag" / "results"
ASR = ROOT / "hw1-asr"
MISSING, DIFFS, COUNT = [], [], [0]


class Missing(Exception):
    pass


def path(base, rel):
    p = base / rel
    if not p.exists():
        if str(p.relative_to(ROOT)) not in MISSING:
            MISSING.append(str(p.relative_to(ROOT)))
        raise Missing(rel)
    return p


def jsonl(rel):
    with path(RAG, rel).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def section(title):
    print(f"\n=== {title} ===")


def block(fn):
    """Run one group of checks; a missing file skips the group, not the script."""
    try:
        fn()
    except Missing as e:
        print(f"  MISSING file: {e}  (the numbers in this group cannot be checked)")


def check(label, got, want, src, tol=None, fmt="{:.3f}"):
    """Compare a recomputed value with the value quoted in the report."""
    COUNT[0] += 1
    if isinstance(want, str) or isinstance(got, str):
        ok = str(got) == str(want)
        g, w = str(got), str(want)
    else:
        if tol is None:
            tol = 0.5 * 10 ** (-int(re.search(r"\.(\d+)f", fmt).group(1))) + 1e-9
        ok = abs(got - want) <= tol
        g, w = fmt.format(got), fmt.format(want)
    flag = "OK  " if ok else "DIFF"
    if not ok:
        DIFFS.append(label)
    print(f"  {flag} {label}: {g}  (report: {w})  [{src}]")


def se(x):
    return stdev(x) / math.sqrt(len(x))


def paired(a, b, qs):
    return [a[q] - b[q] for q in qs]


def bootstrap_ci(x, seed=0, draws=20000):
    rng = random.Random(seed)
    n = len(x)
    means = sorted(mean(rng.choices(x, k=n)) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


# ----------------------------------------------------------------------------
# RAG: Development split
# ----------------------------------------------------------------------------
def load_dev():
    sc = {}
    for c in ("k2", "k10", "k20", "k5b"):
        sc[c] = {r["query_id"]: r["normalised_answer_score"]
                 for r in jsonl(f"phase3/{c}/judge/answer_judge_per_query.jsonl")}
    reps = {}
    for r in jsonl("phase3/k5/judge/judge_cache.jsonl"):
        x = r["result"]
        reps.setdefault(r["query_id"], []).append(
            (x["correctness"] + x["grounding"] + x["completeness"]) / 12)
    sc["k5"] = {q: v[0] for q, v in reps.items()}  # reference: judge repeat 1
    for name, folder in (("hybrid", "hybrid_dev_judge_full125"),
                         ("minilm", "minilm_dev_judge_full125"),
                         ("gold_first", "phase4/gold_first/judge"),
                         ("gold_last", "phase4/gold_last/judge")):
        sc[name] = {r["query_id"]: r["normalised_answer_score"]
                    for r in jsonl(f"{folder}/answer_judge_per_query.jsonl")}
    shared = set.intersection(*[set(sc[c]) for c in
                                ("k2", "k5", "k10", "k20", "k5b", "hybrid", "minilm")])
    ret = {r["query_id"]: r for r in jsonl("retrieval_k_dev.jsonl")}
    return sc, reps, shared, ret


def rag_dev():
    sc, reps, shared, ret = load_dev()
    S = sorted(shared)
    oracle = sorted(shared & set(sc["gold_first"]) & set(sc["gold_last"]))
    nrel = lambda q, k: round(k * ret[q][f"Precision@{k}"])
    src3 = "rag/results/phase3/*/judge"

    section("Sample sizes (Section 3)")
    check("Development queries", len(ret), 125, "rag/results/retrieval_k_dev.jsonl", fmt="{:.0f}")
    check("paired answers in every configuration", len(S), 123, src3, fmt="{:.0f}")
    check("oracle scored queries", len(oracle), 121, "rag/results/phase4", fmt="{:.0f}")
    check("Recall@100 of the dense pool", mean(r["Recall@100"] for r in ret.values()), 0.724,
          "rag/results/retrieval_k_dev.jsonl")
    check("Recall@100 over Recall@5 ('nearly double')", mean(r["Recall@100"] for r in ret.values())
          / mean(r["Recall@5"] for r in ret.values()), 1.9, "rag/results/retrieval_k_dev.jsonl", fmt="{:.1f}")

    section("Table 3 and Figure 2: depth and reranking on Development")
    for k, want in ((2, .279), (5, .379), (10, .475), (20, .580)):
        check(f"Recall@{k}", mean(r[f"Recall@{k}"] for r in ret.values()), want,
              "rag/results/retrieval_k_dev.jsonl")
    want_tok = {"k2": (469, 140, 2271), "k5": (1146, 258, 4269), "k10": (2372, 301, 5149),
                "k20": (4850, 331, 5713), "k5b": (1146, 279, 4644)}
    for c, (wp, wa, wg) in want_tok.items():
        rows = [r for r in jsonl(f"phase3/{c}/answers.jsonl")[1:] if r["query_id"] in shared]
        s = f"rag/results/phase3/{c}/answers.jsonl (first row dropped as warm-up)"
        check(f"{c} median prompt tokens", median(r["prompt_tokens"] for r in rows), wp, s, fmt="{:.0f}")
        check(f"{c} median answer tokens", median(r["answer_tokens"] for r in rows), wa, s, fmt="{:.0f}")
        check(f"{c} median generation ms", median(r["timing"]["generation_ms"] for r in rows), wg, s, fmt="{:.0f}")
    for name, f, wp, wa in (("Hybrid", "hybrid_dev_answers_full125.jsonl", 1184, 260),
                            ("MiniLM", "minilm_dev_answers_full125.jsonl", 1367, 252)):
        rows = [r for r in jsonl(f) if r["query_id"] in shared]
        check(f"{name} median prompt tokens", median(r["prompt_tokens"] for r in rows), wp, f"rag/results/{f}", fmt="{:.0f}")
        check(f"{name} median answer tokens", median(r["answer_tokens"] for r in rows), wa, f"rag/results/{f}", fmt="{:.0f}")
    deltas = {"k2": (.003, .024), "k10": (.020, .025), "k20": (.008, .025), "k5b": (-.004, .022),
              "hybrid": (.014, .022), "minilm": (-.003, .027)}
    for c, (wm, ws) in deltas.items():
        d = paired(sc[c], sc["k5"], S)
        check(f"{c} minus k5: mean", mean(d), wm, src3)
        check(f"{c} minus k5: SE", se(d), ws, src3)
        check(f"{c} minus k5: 95% half-width (Figure 2)", 1.96 * se(d),
              {"k2": .0463, "k10": .0482, "k20": .0489, "k5b": .0422, "hybrid": .0423, "minilm": .0537}[c],
              src3, fmt="{:.4f}")
    d = paired(sc["k20"], sc["k5"], S)
    check("k20 minus k5: 95% interval low", mean(d) - 1.96 * se(d), -0.041, src3)
    check("k20 minus k5: 95% interval high", mean(d) + 1.96 * se(d), 0.057, src3)
    check("baseline mean score", mean(sc["k5"][q] for q in S), 0.327, src3)
    j = [reps[q][1] - reps[q][0] for q in S]
    check("rejudging the same k5 answers: mean", mean(j), -0.010, "rag/results/phase3/k5/judge/judge_cache.jsonl")
    check("rejudging the same k5 answers: SE", se(j), 0.012, "rag/results/phase3/k5/judge/judge_cache.jsonl")
    check("smallest detectable effect, 2.8 x SE of 0.025", 2.8 * 0.025, 0.07, "arithmetic", fmt="{:.2f}")

    section("Oracle (Figure 3) and baseline score by evidence")
    groups = {}
    for q in S:
        groups.setdefault(min(nrel(q, 5), 3), []).append(sc["k5"][q])
    for n, want in ((0, .136), (1, .434), (2, .479), (3, .625)):
        check(f"k5 mean score with {n}{'+' if n == 3 else ''} relevant in top five (n={len(groups[n])})",
              mean(groups[n]), want, src3)
    for arm, wm, ws in (("gold_first", .167, .032), ("gold_last", .118, .031)):
        d = paired(sc[arm], sc["k5"], oracle)
        check(f"{arm} minus k5: mean", mean(d), wm, "rag/results/phase4")
        check(f"{arm} minus k5: SE", se(d), ws, "rag/results/phase4")
        check(f"{arm} minus k5: 95% half-width (Figure 3)", 1.96 * se(d),
              {"gold_first": .063, "gold_last": .061}[arm], "rag/results/phase4", tol=0.0011)
    d = paired(sc["gold_first"], sc["k5"], oracle)
    check("oracle gain in standard errors", mean(d) / se(d), 5.2, "rag/results/phase4", fmt="{:.1f}")
    had0 = [q for q in oracle if nrel(q, 5) == 0]
    had1 = [q for q in oracle if nrel(q, 5) >= 1]
    for lab, qs, wn, wm, ws, wh in (("no relevant record at k=5", had0, 51, .320, .048, .093),
                                    ("at least one at k=5", had1, 70, .056, .038, .075)):
        d = paired(sc["gold_first"], sc["k5"], qs)
        check(f"gold-first, {lab}: n", len(d), wn, "rag/results/phase4", fmt="{:.0f}")
        check(f"gold-first, {lab}: mean", mean(d), wm, "rag/results/phase4")
        check(f"gold-first, {lab}: SE", se(d), ws, "rag/results/phase4")
        check(f"gold-first, {lab}: 95% half-width (Figure 3)", 1.96 * se(d), wh, "rag/results/phase4", tol=0.0011)

    section("Order effect")
    ids = {c: {r["query_id"]: r["retrieved_record_ids"] for r in jsonl(f"phase4/{c}/answers.jsonl")}
           for c in ("gold_first", "gold_last")}
    moved = [q for q in oracle if ids["gold_first"][q] != ids["gold_last"][q]]
    for lab, qs, wn, wm, ws in (("all changed orders", moved, 106, .057, .031),
                                ("no relevant record at k=5", [q for q in moved if nrel(q, 5) == 0], 47, .154, .048),
                                ("at least one at k=5", [q for q in moved if nrel(q, 5) >= 1], 59, -.020, .039)):
        d = paired(sc["gold_first"], sc["gold_last"], qs)
        check(f"gold-first minus gold-last, {lab}: n", len(d), wn, "rag/results/phase4", fmt="{:.0f}")
        check(f"gold-first minus gold-last, {lab}: mean", mean(d), wm, "rag/results/phase4")
        check(f"gold-first minus gold-last, {lab}: SE", se(d), ws, "rag/results/phase4")

    section("How much of the oracle's gain depth realises")
    rescued = [q for q in S if nrel(q, 5) == 0 and nrel(q, 20) >= 1]
    enriched = [q for q in S if nrel(q, 5) >= 1 and nrel(q, 20) > nrel(q, 5)]
    check("queries given a first labelled record by k=20", len(rescued), 23, src3, fmt="{:.0f}")
    check("queries given further labelled records by k=20", len(enriched), 30, src3, fmt="{:.0f}")
    gained = [q for q in rescued + enriched if q in sc["gold_last"] and q in sc["gold_first"]]
    lo = sum(paired(sc["gold_last"], sc["k5"], gained)) / len(S)
    hi = sum(paired(sc["gold_first"], sc["k5"], gained)) / len(S)
    check("ceiling, low end (gold-last gains on those queries / 123)", lo, 0.08, "rag/results/phase4", fmt="{:.2f}")
    check("ceiling, high end (gold-first gains on those queries / 123)", hi, 0.09, "rag/results/phase4", fmt="{:.2f}")
    check("k20 minus k5 on those 53: mean", mean(paired(sc["k20"], sc["k5"], gained)), 0.039, src3)
    check("k20 minus k5 on those 53: SE", se(paired(sc["k20"], sc["k5"], gained)), 0.041, src3)
    check("gold-last minus k5 on those 53: mean", mean(paired(sc["gold_last"], sc["k5"], gained)), 0.179, "rag/results/phase4")
    check("gold-last minus k5 on those 53: SE", se(paired(sc["gold_last"], sc["k5"], gained)), 0.046, "rag/results/phase4")
    d = paired(sc["gold_last"], sc["k20"], gained)
    check("gold-last minus k20 on those 53: mean", mean(d), 0.140, "rag/results/phase4")
    check("gold-last minus k20 on those 53: SE", se(d), 0.039, "rag/results/phase4")
    share = sum(paired(sc["k20"], sc["k5"], gained)) / sum(paired(sc["gold_last"], sc["k5"], gained))
    check("share of the available gain realised ('about a fifth')", share, 0.2, src3, fmt="{:.1f}")
    check("largest possible average gain from one improved query", 1 / len(S), 0.008, "arithmetic")

    section("Depth and response cost")
    none = [q for q in S if nrel(q, 20) == nrel(q, 5)]
    tok = {c: {r["query_id"]: r["answer_tokens"] for r in jsonl(f"phase3/{c}/answers.jsonl")} for c in ("k5", "k20")}
    check("k=20 answer tokens added where no new evidence", mean(paired(tok["k20"], tok["k5"], none)), 40,
          "rag/results/phase3", tol=2, fmt="{:.0f}")
    comp20 = {r["query_id"]: r["completeness"] for r in jsonl("phase3/k20/judge/answer_judge_per_query.jsonl")}
    comp5 = {}
    for r in jsonl("phase3/k5/judge/judge_cache.jsonl"):
        comp5.setdefault(r["query_id"], r["result"]["completeness"])
    d = paired(comp20, comp5, none)
    check("completeness change there: mean (0-4 scale)", mean(d), -0.07, src3, fmt="{:.2f}")
    check("completeness change there: SE", se(d), 0.13, src3, fmt="{:.2f}")
    check("k=20 median generation time above k=5", 5713 / 4269 - 1, 0.34, "Table 3", fmt="{:.2f}")
    check("k5 rerun slower than original", 4644 / 4269 - 1, 0.09, "Table 3", fmt="{:.2f}")
    check("generation share of the 12.7x reranking gap", (280.52 - 22.09) / 4269, 0.06, "Table 3", fmt="{:.2f}")


def rag_rerank():
    sc, reps, shared, ret = load_dev()
    S = sorted(shared)
    section("Rerankers: retrieval changes on Development")
    h = {r["query_id"]: r for r in jsonl("hybrid_dev_alpha095.jsonl")}
    m = {r["query_id"]: r for r in jsonl("minilm_dev_full125.jsonl")}
    hs, msrc = "rag/results/hybrid_dev_alpha095.jsonl", "rag/results/minilm_dev_full125.jsonl"
    check("dense Recall@5", mean(r["dense_metrics"]["Recall@5"] for r in h.values()), 0.3795, hs, fmt="{:.4f}")
    check("Hybrid Recall@5", mean(r["hybrid_metrics"]["Recall@5"] for r in h.values()), 0.3801, hs, fmt="{:.4f}")
    check("Hybrid reorders the top five on", sum(r["retrieved_record_ids"] != r["dense_record_ids"][:5] for r in h.values()), 70, hs, fmt="{:.0f}")
    check("Hybrid changes top-five membership on", sum(set(r["retrieved_record_ids"]) != set(r["dense_record_ids"][:5]) for r in h.values()), 24, hs, fmt="{:.0f}")
    check("Hybrid improves labelled recall on", sum(r["hybrid_metrics"]["Recall@5"] > r["dense_metrics"]["Recall@5"] + 1e-9 for r in h.values()), 1, hs, fmt="{:.0f}")
    check("MiniLM Recall@5", mean(r["minilm_metrics"]["Recall@5"] for r in m.values()), 0.3326, msrc, fmt="{:.4f}")
    dm = {q: r["minilm_metrics"]["Recall@5"] - r["dense_metrics"]["Recall@5"] for q, r in m.items()}
    check("MiniLM recall gains (of 125)", sum(v > 1e-9 for v in dm.values()), 14, msrc, fmt="{:.0f}")
    check("MiniLM recall losses (of 125)", sum(v < -1e-9 for v in dm.values()), 24, msrc, fmt="{:.0f}")
    check("MiniLM changes top-five membership on", sum(set(r["retrieved_record_ids"]) != set(r["dense_record_ids"][:5]) for r in m.values()), 124, msrc, fmt="{:.0f}")

    section("Rerankers: answer quality subgroups")
    lost = [q for q in S if dm[q] < -1e-9]
    gain = [q for q in S if dm[q] > 1e-9]
    d = paired(sc["minilm"], sc["k5"], lost)
    src = "rag/results/minilm_dev_judge_full125"
    check("MiniLM recall-loss queries judged", len(d), 23, src, fmt="{:.0f}")
    check("MiniLM recall-loss: mean", mean(d), -0.167, src)
    check("MiniLM recall-loss: SE", se(d), 0.056, src)
    lo, hi = bootstrap_ci(d)
    check("MiniLM recall-loss: bootstrap 95% low (seed 0, 20,000 draws)", lo, -0.275, src, tol=0.006)
    check("MiniLM recall-loss: bootstrap 95% high", hi, -0.062, src, tol=0.006)
    d = paired(sc["minilm"], sc["k5"], gain)
    check("MiniLM recall-gain queries judged", len(d), 14, src, fmt="{:.0f}")
    check("MiniLM recall-gain vs k5: mean", mean(d), -0.042, src)
    check("MiniLM recall-gain vs k5: SE", se(d), 0.117, src)
    check("MiniLM recall-gain vs the k5 rerun: mean", mean(paired(sc["minilm"], sc["k5b"], gain)), 0.107, src)
    ha = {r["query_id"]: r for r in jsonl("hybrid_dev_answers_full125.jsonl")}
    reused = [q for q in S if ha[q].get("answer_origin") == "reused_dense_k5b_identical_prompt"]
    changed = [q for q in S if q not in set(reused)]
    src = "rag/results/hybrid_dev_answers_full125.jsonl + hybrid_dev_judge_full125"
    check("Hybrid answers reused from the k5 rerun", len(reused), 56, src, fmt="{:.0f}")
    check("Hybrid minus k5 on reused answers: mean", mean(paired(sc["hybrid"], sc["k5"], reused)), 0.052, src)
    check("Hybrid queries with a changed context", len(changed), 67, src, fmt="{:.0f}")
    check("Hybrid minus k5 on changed contexts: mean", mean(paired(sc["hybrid"], sc["k5"], changed)), -0.017, src)
    check("Hybrid minus k5 on changed contexts: SE", se(paired(sc["hybrid"], sc["k5"], changed)), 0.026, src)
    uk = [r for r in m.values() if "self" in r["transcript"].lower() and "minor" in r["transcript"].lower()]
    if uk:
        q = uk[0]["query_id"]
        check("illustrative case: k5 score", reps[q][0], 1.000, "rag/results/phase3/k5/judge/judge_cache.jsonl")
        check("illustrative case: MiniLM score", sc["minilm"][q], 0.417, "rag/results/minilm_dev_judge_full125")
    else:
        print("  NOTE: illustrative UK self-employment query not found by keyword")


def rag_practice_tuning():
    section("Practice split: reranker tuning")
    with path(RAG, "hybrid_alpha_sweep_practice.csv").open() as f:
        rows = {float(r["alpha"]): r for r in csv.DictReader(f)}
    src = "rag/results/hybrid_alpha_sweep_practice.csv"
    check("alpha values in the sweep", len(rows), 10, src, fmt="{:.0f}")
    check("BM25 alone (alpha 0): Recall@5", float(rows[0.0]["Recall@5"]), 0.288, src)
    check("equal weighting (alpha 0.5): Recall@5", float(rows[0.5]["Recall@5"]), 0.384, src)
    check("alpha 0.95: queries whose Recall@5 changed", int(rows[0.95]["improved"]) + int(rows[0.95]["worsened"]), 3, src, fmt="{:.0f}")
    best = max(rows, key=lambda a: float(rows[a]["Recall@5"]))
    check("alpha with the highest Practice Recall@5", best, 0.95, src, fmt="{:.2f}")


def rag_practice_minilm():
    section("Practice split: MiniLM retrieval")
    m = jsonl("minilm_practice_full350.jsonl")
    src = "rag/results/minilm_practice_full350.jsonl"
    check("Practice queries", len(m), 350, src, fmt="{:.0f}")
    check("dense Practice Recall@5", mean(r["dense_metrics"]["Recall@5"] for r in m), 0.421, src)
    check("MiniLM Practice Recall@5", mean(r["minilm_metrics"]["Recall@5"] for r in m), 0.388, src)


def rag_cost():
    section("Matched reranking cost")
    files = sorted(glob.glob(str(RAG / "reranking_matched_*.json")))
    if not files:
        MISSING.append("rag/results/reranking_matched_*.json")
        raise Missing("reranking_matched_*.json")
    text = Path(files[0]).read_text()
    src = "rag/" + str(Path(files[0]).relative_to(ROOT / "rag"))
    for label, value in (("Hybrid mean ms", "22.09"), ("Hybrid median ms", "22.35"),
                         ("MiniLM mean ms", "280.52"), ("MiniLM median ms", "281.84"),
                         ("calls per method", "369"),
                         ("BM25 setup s", "3.52"), ("MiniLM load s", "18.67")):
        found = [x for x in re.findall(r"-?\d+\.?\d*", text)
                 if abs(float(x) - float(value)) < (0.006 if "." in value else 0.5)
                 or abs(float(x) / 1000 - float(value)) < 0.006]
        check(label, value if found else "not found", value, src)
    check("MiniLM / Hybrid mean time", 280.52 / 22.09, 12.70, "arithmetic", fmt="{:.2f}")


def rag_practice_depth():
    section("Practice split: pre-specified depth replication")
    P = "practice_depth"
    sc = {a: {r["query_id"]: r["normalised_answer_score"]
              for r in jsonl(f"{P}/{a}/judge/answer_judge_per_query.jsonl")} for a in ("k5", "k2", "k20")}
    src = "rag/results/practice_depth/*/judge"
    excl = []
    for a in ("k5", "k2", "k20"):
        p = RAG / P / a / "judge" / "excluded.jsonl"
        excl.append(sum(1 for line in p.open() if line.strip()) if p.exists() else 0)
    check("judge exclusions for k5, k2, k20", f"{excl[0]}, {excl[1]}, {excl[2]}", "2, 1, 0", src)
    d2 = paired(sc["k2"], sc["k5"], sorted(set(sc["k2"]) & set(sc["k5"])))
    d20 = paired(sc["k20"], sc["k5"], sorted(set(sc["k20"]) & set(sc["k5"])))
    check("k2 minus k5: n", len(d2), 347, src, fmt="{:.0f}")
    check("k2 minus k5: mean", mean(d2), -0.008, src)
    check("k2 minus k5: SE", se(d2), 0.016, src)
    check("k2 minus k5: one-sided 95% bound", mean(d2) - 1.645 * se(d2), -0.034, src)
    check("k20 minus k5: mean", mean(d20), -0.020, src)
    check("k20 minus k5: SE", se(d20), 0.017, src)
    dsc, _, shared, _ = load_dev()
    S = sorted(shared)
    p2 = d2 + paired(dsc["k2"], dsc["k5"], S)
    p20 = d20 + paired(dsc["k20"], dsc["k5"], S)
    src = "rag/results/practice_depth + rag/results/phase3"
    check("pooled k2 minus k5: n", len(p2), 470, src, fmt="{:.0f}")
    check("pooled k2 minus k5: mean", mean(p2), -0.005, src)
    check("pooled k2 minus k5: SE", se(p2), 0.013, src)
    check("pooled k2 minus k5: one-sided 95% bound", mean(p2) - 1.645 * se(p2), -0.027, src)
    check("pooled k20 minus k5: mean", mean(p20), -0.013, src)
    check("pooled k20 minus k5: SE", se(p20), 0.014, src)
    check("pooled k20 minus k5: 95% interval low", mean(p20) - 1.96 * se(p20), -0.040, src)
    check("pooled k20 minus k5: 95% interval high", mean(p20) + 1.96 * se(p20), 0.014, src)
    check("pooled k20 minus k5: n", len(p20), 471, src, fmt="{:.0f}")
    check("k2 minus k5: 95% half-width (Figure 2)", 1.96 * se(d2), 0.0313, "rag/results/practice_depth", fmt="{:.4f}")
    check("k20 minus k5: 95% half-width (Figure 2)", 1.96 * se(d20), 0.0328, "rag/results/practice_depth", fmt="{:.4f}")
    check("pooled k2: 95% half-width (Figure 2)", 1.96 * se(p2), 0.0261, src, fmt="{:.4f}")
    check("pooled k20: 95% half-width (Figure 2)", 1.96 * se(p20), 0.0274, src, fmt="{:.4f}")
    pret = jsonl("retrieval_k_practice.jsonl")
    for k, want in ((2, .319), (5, .421), (20, .579)):
        check(f"Practice Recall@{k} (results table)", mean(r[f"Recall@{k}"] for r in pret), want,
              "rag/results/retrieval_k_practice.jsonl")
    med = {}
    for a, want, wp, wa in (("k2", 2575, 497, 144), ("k5", 4028, 1212, 223), ("k20", 4607, 4993, 244)):
        rows = jsonl(f"{P}/{a}/answers.jsonl")
        med[a] = median(r["timing"]["generation_ms"] for r in rows)
        s_ = f"rag/results/practice_depth/{a}/answers.jsonl"
        check(f"{a} median generation ms", med[a], want, s_, fmt="{:.0f}")
        check(f"{a} median prompt tokens", median(r["prompt_tokens"] for r in rows), wp, s_, tol=0.6, fmt="{:.0f}")
        check(f"{a} median answer tokens", median(r["answer_tokens"] for r in rows), wa, s_, tol=0.6, fmt="{:.0f}")
    check("k=2 time saving on Practice", 1 - med["k2"] / med["k5"], 0.36, "practice_depth answers", fmt="{:.2f}")
    check("k=2 time saving on Development", 1 - 2271 / 4269, 0.47, "Table 3", fmt="{:.2f}")


def ols_latency(rows):
    """Least squares: generation_ms = b0 + b1 * answer_tokens + b2 * prompt_tokens."""
    y = [r["timing"]["generation_ms"] for r in rows]
    X = [[1.0, r["answer_tokens"], r["prompt_tokens"]] for r in rows]
    M = [[sum(x[i] * x[j] for x in X) for j in range(3)] + [sum(x[i] * v for x, v in zip(X, y))]
         for i in range(3)]
    for i in range(3):
        piv = M[i][i]
        M[i] = [v / piv for v in M[i]]
        for j in range(3):
            if j != i:
                f = M[j][i]
                M[j] = [a - f * b for a, b in zip(M[j], M[i])]
    b = [M[i][3] for i in range(3)]
    pred = [b[0] + b[1] * x[1] + b[2] * x[2] for x in X]
    my = mean(y)
    r2 = 1 - sum((u - v) ** 2 for u, v in zip(y, pred)) / sum((u - my) ** 2 for u in y)
    return b, r2


def rag_followup_saved():
    """Follow-up checks on saved data (no new runs): latency model, k=2 and k=20 subgroups,
    judge stability. Reference values are the figures proposed for the report."""
    section("Follow-up on saved data: what generation time depends on")
    dev = [r for c in ("k2", "k5", "k5b", "k10", "k20") for r in jsonl(f"phase3/{c}/answers.jsonl")[1:]]
    pra = [r for c in ("k2", "k5", "k20") for r in jsonl(f"practice_depth/{c}/answers.jsonl")[1:]]
    for name, rows, want in (("Development", dev, (16.1, 0.09, 0.99)), ("Practice", pra, (17.0, 0.06, 0.96))):
        b, r2 = ols_latency(rows)
        s = "rag/results/phase3/*/answers.jsonl" if name == "Development" else "rag/results/practice_depth/*/answers.jsonl"
        check(f"{name}: ms per answer token (n={len(rows)}, warm-up rows dropped)", b[1], want[0], s, fmt="{:.1f}")
        check(f"{name}: ms per prompt token", b[2], want[1], s, fmt="{:.2f}")
        check(f"{name}: R^2", r2, want[2], s, fmt="{:.2f}")

    section("Follow-up on saved data: depth subgroups by labelled evidence")
    dsc, reps, shared, dret = load_dev()
    psc = {a: {r["query_id"]: r["normalised_answer_score"]
               for r in jsonl(f"practice_depth/{a}/judge/answer_judge_per_query.jsonl")} for a in ("k5", "k2", "k20")}
    pret = {r["query_id"]: r for r in jsonl("retrieval_k_practice.jsonl")}
    nrel = lambda ret, q, k: round(k * ret[q][f"Precision@{k}"])
    for split, sc, ret, base, src in (
            ("Development", dsc, dret, sorted(shared), "rag/results/phase3 + retrieval_k_dev.jsonl"),
            ("Practice", psc, pret, None, "rag/results/practice_depth + retrieval_k_practice.jsonl")):
        want2 = {"Development": ((16, -.130, .071), (107, .023, .025)),
                 "Practice": ((35, -.174, .053), (312, .011, .016))}[split]
        qs = base or sorted(set(sc["k2"]) & set(sc["k5"]))
        drop = [q for q in qs if nrel(ret, q, 5) > 0 and nrel(ret, q, 2) == 0]
        rest = [q for q in qs if q not in drop]
        for lab, g, (wn, wm, ws) in (("k=2 drops every gold record k=5 had", drop, want2[0]),
                                     ("elsewhere", rest, want2[1])):
            d = paired(sc["k2"], sc["k5"], g)
            check(f"{split}, k2 minus k5, {lab}: n", len(d), wn, src, fmt="{:.0f}")
            check(f"{split}, k2 minus k5, {lab}: mean", mean(d), wm, src)
            check(f"{split}, k2 minus k5, {lab}: SE", se(d), ws, src)
        want20 = {"Development": ((23, .083, .043), (30, .006, .065), (70, -.015, .031)),
                  "Practice": ((51, .043, .045), (58, -.024, .050), (239, -.033, .019))}[split]
        qs = base or sorted(set(sc["k20"]) & set(sc["k5"]))
        groups = (("supplies the first gold record", [q for q in qs if nrel(ret, q, 5) == 0 and nrel(ret, q, 20) > 0]),
                  ("adds further gold", [q for q in qs if nrel(ret, q, 5) > 0 and nrel(ret, q, 20) > nrel(ret, q, 5)]),
                  ("adds no gold", [q for q in qs if nrel(ret, q, 20) == nrel(ret, q, 5)]))
        for (lab, g), (wn, wm, ws) in zip(groups, want20):
            d = paired(sc["k20"], sc["k5"], g)
            check(f"{split}, k20 minus k5, {lab}: n", len(d), wn, src, fmt="{:.0f}")
            check(f"{split}, k20 minus k5, {lab}: mean", mean(d), wm, src, tol=0.0011)
            check(f"{split}, k20 minus k5, {lab}: SE", se(d), ws, src)

    section("Follow-up on saved data: judge stability over three judgements of the same k5 answers")
    tot = {}
    for r in jsonl("phase3/k5/judge/judge_cache.jsonl"):
        x = r["result"]
        tot.setdefault(r["query_id"], []).append(x["correctness"] + x["grounding"] + x["completeness"])
    spread = [max(v[:3]) - min(v[:3]) for v in tot.values() if len(v) >= 3]
    src = "rag/results/phase3/k5/judge/judge_cache.jsonl"
    check("queries with three judgements", len(spread), 125, src, fmt="{:.0f}")
    check("queries whose three totals are not all equal", sum(s > 0 for s in spread), 28, src, fmt="{:.0f}")
    check("queries whose totals span 3+ points of 12", sum(s >= 3 for s in spread), 13, src, fmt="{:.0f}")


# ----------------------------------------------------------------------------
# ASR: values traced to saved logs
# ----------------------------------------------------------------------------
def log(rel):
    return path(ASR, rel).read_text(errors="replace").splitlines()


def grab(rel, line_re, value_re, which=0):
    """First line matching line_re; return the number captured by value_re."""
    for line in log(rel):
        if re.search(line_re, line):
            found = re.findall(value_re, line)
            if found:
                return float(found[which])
    return "not found"


TIME = r"Time:\s+([\d.]+) ms per transcription"
SPREAD = r"\(\+/- ([\d.]+) ms\)"
PERTOK = r"([\d.]+) ms per token"
GBS = r"([\d.]+) GB/s"
GFLOPS = r"(\d+) GFLOP/s"
MS = r"([\d.]+) ms"
ERR = r"max_err=([\d.e+-]+)"


def asr_table():
    section("ASR: transcription latency table")
    rows = (("PyTorch", "results/baseline_profile.txt", 768.9, 1.1, 59.1),
            ("RoPE + reductions custom", "results/task2_model.txt", 729.0, 0.3, 56.1),
            ("All custom, original GEMM", "results/final_profile.txt", 2689.8, 5.6, 206.9),
            ("GEMM v2, fixed 64x64", "extensions/results/v2_e2e_64x64_pad.txt", 1644.7, 2.0, 126.5),
            ("GEMM v2, shape-aware", "extensions/results/v2_e2e_dispatch.txt", 1563.6, 2.4, 120.3))
    for name, f, t, s, tok in rows:
        check(f"{name}: ms", grab(f, "Time:", TIME), t, f"hw1-asr/{f}", fmt="{:.1f}")
        check(f"{name}: spread", grab(f, "Time:", SPREAD), s, f"hw1-asr/{f}", fmt="{:.1f}")
        check(f"{name}: ms per token", grab(f, "Time:", PERTOK), tok, f"hw1-asr/{f}", fmt="{:.1f}")
    check("PyTorch baseline rerun alongside the final kernels", grab("results/final_baseline_profile.txt", "Time:", TIME),
          766.9, "hw1-asr/results/final_baseline_profile.txt", fmt="{:.1f}")
    check("original backend / PyTorch ('about 3.5x')", 2689.8 / 768.9, 3.5, "arithmetic", fmt="{:.1f}")
    check("GEMM v2 / PyTorch", 1563.6 / 768.9, 2.03, "arithmetic", fmt="{:.2f}")
    check("original / GEMM v2", 2689.8 / 1563.6, 1.72, "arithmetic", fmt="{:.2f}")
    check("reduction in custom-backend time", 1 - 1563.6 / 2689.8, 0.42, "arithmetic", fmt="{:.2f}")
    lines = log("results/final_test.txt")
    check("correctness cases passing", sum(bool(re.search(r"PASS\s+max_err", x)) for x in lines), 33,
          "hw1-asr/results/final_test.txt", fmt="{:.0f}")
    check("tasks reporting ALL PASS", sum("ALL PASS" in x for x in lines), 4, "hw1-asr/results/final_test.txt", fmt="{:.0f}")
    check("clip length s", grab("results/final_profile.txt", "Audio:", r"Audio: ([\d.]+) s"), 3.5, "hw1-asr/results/final_profile.txt", fmt="{:.1f}")
    check("prompt tokens", grab("results/final_profile.txt", "Audio:", r"prompt (\d+) tokens"), 59, "hw1-asr/results/final_profile.txt", fmt="{:.0f}")
    check("generated tokens", grab("results/final_profile.txt", "run 1:", r"\((\d+) tokens\)"), 13, "hw1-asr/results/final_profile.txt", fmt="{:.0f}")


def asr_rope_reduce():
    section("ASR: RoPE and reductions")
    enc = r"audio encoder \(partial"
    vals = [grab(f"results/task1_threads{t}.txt", enc, GBS) for t in (64, 128, 256, 512)]
    print(f"  INFO RoPE encoder GB/s at 64-512 threads: {vals}  (report: 'about 490', 79-80% of 616)")
    check("RoPE encoder GB/s (final test)", grab("results/final_test.txt", enc, GBS), 490, "hw1-asr/results/final_test.txt", tol=3, fmt="{:.0f}")
    check("share of 616 GB/s peak, lowest", min(vals) / 616, 0.79, "hw1-asr/results/task1_threads*.txt", fmt="{:.2f}")
    check("share of 616 GB/s peak, highest", max(vals) / 616, 0.80, "hw1-asr/results/task1_threads*.txt", fmt="{:.2f}")
    check("accurate build, encoder error", grab("results/final_test.txt", enc, ERR), 1.93e-7, "hw1-asr/results/final_test.txt", tol=1e-9, fmt="{:.2e}")
    check("fast-math build, encoder error", grab("results/task1_fastmath.txt", enc, ERR), 1.94e-4, "hw1-asr/results/task1_fastmath.txt", tol=1e-6, fmt="{:.2e}")
    check("fast-math build, decoder error (still passes)", grab("results/task1_fastmath.txt", r"text decoder", ERR), 9.1e-6, "hw1-asr/results/task1_fastmath.txt", tol=1e-7, fmt="{:.1e}")
    ln = r"layernorm, audio encoder"
    lv = [grab(f, ln, GBS) for f in ("results/task2_test.txt", "results/task2_threads256.txt", "results/final_test.txt")]
    check("LayerNorm encoder GB/s, lowest run", min(lv), 355, "hw1-asr/results/task2_*.txt, final_test.txt", tol=1, fmt="{:.0f}")
    check("LayerNorm encoder GB/s, highest run", max(lv), 380, "hw1-asr/results/task2_*.txt, final_test.txt", tol=1, fmt="{:.0f}")
    check("vocabulary softmax GB/s", grab("results/final_test.txt", r"softmax, vocabulary row", GBS), 3.6, "hw1-asr/results/final_test.txt", fmt="{:.1f}")
    v1024 = grab("results/task2_threads1024.txt", r"softmax, vocabulary row", GBS)
    l1024 = grab("results/task2_threads1024.txt", ln, GBS)
    print(f"  INFO at 1024 threads: vocabulary softmax {v1024} GB/s (helps), LayerNorm {l1024} GB/s (hurts)")
    text = "\n".join(log("results/task2_nobarrier_racecheck.txt"))
    check("Racecheck hazards without the barrier", float(re.search(r"\[(\d+) hazards\]", text).group(1)), 160,
          "hw1-asr/results/task2_nobarrier_racecheck.txt", fmt="{:.0f}")
    mm = re.search(r"(\d+) of (\d+) checked values are wrong", text)
    check("wrong values under the sanitizer", f"{mm.group(1)} of {mm.group(2)}", "3904 of 5005",
          "hw1-asr/results/task2_nobarrier_racecheck.txt")
    check("ordinary reduction tests still passing without the barrier",
          sum(bool(re.search(r"PASS\s+max_err", x)) for x in log("results/task2_nobarrier_test.txt")), 11,
          "hw1-asr/results/task2_nobarrier_test.txt", fmt="{:.0f}")
    check("decoder RMSNorm ms with shuffles", grab("results/task2_test.txt", r"rmsnorm, text decoder", MS), 0.006, "hw1-asr/results/task2_test.txt")
    check("decoder RMSNorm ms with atomics", grab("results/task2_atomic.txt", r"rmsnorm, text decoder", MS), 0.022, "hw1-asr/results/task2_atomic.txt")


def asr_gemm():
    section("ASR: first GEMM")
    eq = r"encoder q_proj"
    check("encoder q_proj GFLOP/s, T=8", grab("results/task3_tile8.txt", eq, GFLOPS), 1168, "hw1-asr/results/task3_tile8.txt", fmt="{:.0f}")
    check("encoder q_proj GFLOP/s, T=16", grab("results/task3_test.txt", eq, GFLOPS), 1865, "hw1-asr/results/task3_test.txt", fmt="{:.0f}")
    check("encoder q_proj GFLOP/s, T=32", grab("results/task3_tile32.txt", eq, GFLOPS), 2130, "hw1-asr/results/task3_tile32.txt", fmt="{:.0f}")
    lines = log("results/task3_tile32.txt")
    i = next(n for n, x in enumerate(lines) if re.search(eq, x))
    check("cuBLAS on encoder q_proj", float(re.search(r"cuBLAS\s+(\d+) GFLOP/s", lines[i + 1]).group(1)), 13164,
          "hw1-asr/results/task3_tile32.txt", fmt="{:.0f}")
    check("no weight padding, T=16", grab("results/task3_nopad.txt", eq, GFLOPS), 830, "hw1-asr/results/task3_nopad.txt", fmt="{:.0f}")
    check("padding cost", 1865 / 830, 2.25, "arithmetic", fmt="{:.2f}")
    check("uncoalesced loads, T=16", grab("results/task3_uncoalesced.txt", eq, GFLOPS), 1350, "hw1-asr/results/task3_uncoalesced.txt", fmt="{:.0f}")
    check("LM head GFLOP/s, T=8", grab("results/task3_tile8.txt", r"lm_head", GFLOPS), 159, "hw1-asr/results/task3_tile8.txt", fmt="{:.0f}")
    check("LM head GFLOP/s, T=32", grab("results/task3_tile32.txt", r"lm_head", GFLOPS), 63, "hw1-asr/results/task3_tile32.txt", fmt="{:.0f}")
    fails = sum(bool(re.search(r"FAIL\s+max_err", x)) for x in log("results/task3_nobarrier.txt"))
    check("wrong results without the second barrier", "yes" if fails else "no", "yes",
          f"hw1-asr/results/task3_nobarrier.txt ({fails} cases fail)")
    f = "extensions/results/v1_profile_by_M.txt"
    calls_m1 = grab(f, r"^\s+linear_m1\s", r"\s(\d+)\s+[\d.]+\s+[\d.]+%")
    calls = grab(f, r"^\s+linear\s", r"\s(\d+)\s+[\d.]+\s+[\d.]+%")
    t_m1 = grab(f, r"^\s+linear_m1\s", r"\s([\d.]+)\s+[\d.]+%")
    t = grab(f, r"^\s+linear\s", r"\s([\d.]+)\s+[\d.]+%")
    check("GEMM calls with M=1", calls_m1, 13, f"hw1-asr/{f}", fmt="{:.0f}")
    check("GEMM calls in total", calls + calls_m1, 2757, f"hw1-asr/{f}", fmt="{:.0f}")
    check("share of GEMM time with M=1", t_m1 / (t + t_m1), 0.020, f"hw1-asr/{f}")
    check("fused GELU traffic saved, MB", 1500 * 5120 * 4 * 2 / 1e6, 61.44, "arithmetic", fmt="{:.2f}")


def asr_attention():
    section("ASR: attention")
    enc = r"audio encoder \(20 heads"
    check("score matrix, MB", 20 * 1500 ** 2 * 4 / 1e6, 180, "arithmetic", fmt="{:.0f}")
    check("shared memory per block, bytes", 4 * (32 * 65 + 32 * 64 + 16 * 64), 20608, "arithmetic", fmt="{:.0f}")
    check("key loads per head at 4 warps", math.ceil(1500 / 4), 375, "arithmetic", fmt="{:.0f}")
    check("key loads per head at 16 warps", math.ceil(1500 / 16), 94, "arithmetic", fmt="{:.0f}")
    check("encoder ms at 1 warp", grab("results/task4_warps1.txt", enc, MS), 122, "hw1-asr/results/task4_warps1.txt", tol=0.5, fmt="{:.0f}")
    check("encoder ms at 16 warps", grab("results/task4_warps16.txt", enc, MS), 10.6, "hw1-asr/results/task4_warps16.txt", fmt="{:.1f}")
    check("encoder ms at 4 warps, padded", grab("results/task4_test.txt", enc, MS), 17.0, "hw1-asr/results/task4_test.txt", fmt="{:.1f}")
    check("encoder ms at 4 warps, no key padding", grab("results/task4_nopad.txt", enc, MS), 57.8, "hw1-asr/results/task4_nopad.txt", fmt="{:.1f}")
    check("padding cost", 57.8 / 17.0, 3.4, "arithmetic", fmt="{:.1f}")
    att = r"^\s+attention\s"
    val = r"\s([\d.]+)\s+[\d.]+%"
    a1 = grab("results/final_profile.txt", att, val)
    a2 = grab("extensions/results/v2_e2e_dispatch.txt", att, val)
    check("custom attention profiled ms, low", min(a1, a2), 268, "hw1-asr/results/final_profile.txt", tol=0.5, fmt="{:.0f}")
    check("custom attention profiled ms, high", max(a1, a2), 277, "hw1-asr/extensions/results/v2_e2e_dispatch.txt", tol=0.5, fmt="{:.0f}")
    t1 = grab("results/task4_profile_torchattn.txt", att, val)
    t2 = grab("results/baseline_profile.txt", att, val)
    check("PyTorch attention ms, other kernels custom", t1, 131, "hw1-asr/results/task4_profile_torchattn.txt", tol=0.6, fmt="{:.0f}")
    check("PyTorch attention ms, all PyTorch", t2, 237, "hw1-asr/results/baseline_profile.txt", tol=0.6, fmt="{:.0f}")
    pk = r"Peak GPU memory: ([\d.]+) GiB"
    check("peak memory saved, GiB",
          grab("results/final_baseline_profile.txt", "Peak GPU", pk) - grab("results/final_profile.txt", "Peak GPU", pk),
          0.32, "hw1-asr/results/final_*profile.txt", fmt="{:.2f}")


def asr_v2():
    section("ASR: GEMM v2 table and profile")
    shapes = (("Dec. q", r"decoder q_proj"), ("Dec. gate", r"decoder gate_proj"),
              ("Enc. q", r"encoder q_proj"), ("Enc. FC1", r"encoder fc1"))
    table = (("Original tile 32", "v2_test_before.txt", (1440, 1496, 2137, 2111)),
             ("Register 64x64", "v2_regtile_64x64.txt", (1686, 2517, 3844, 3971)),
             ("Padded 32x64", "v2_regtile_32x64_pad.txt", (1999, 3299, 4208, 3721)),
             ("Padded 64x64", "v2_regtile_64x64_pad.txt", (1793, 3085, 5307, 5254)),
             ("Final, tile by shape", "v2_dispatch_test.txt", (1999, 3269, 5304, 5297)))
    for name, f, want in table:
        for (col, pat), w in zip(shapes, want):
            check(f"{name}, {col}", grab(f"extensions/results/{f}", pat, GFLOPS), w,
                  f"hw1-asr/extensions/results/{f}", fmt="{:.0f}")
    lines = log("extensions/results/v2_dispatch_test.txt")
    cub = []
    for (col, pat), w in zip(shapes, (4685, 6696, 12973, 12957)):
        i = next(n for n, x in enumerate(lines) if re.search(pat, x))
        v = float(re.search(r"cuBLAS\s+(\d+) GFLOP/s", lines[i + 1]).group(1))
        cub.append(v)
        check(f"cuBLAS, {col}", v, w, "hw1-asr/extensions/results/v2_dispatch_test.txt", fmt="{:.0f}")
    ratios = [a / b for a, b in zip((1999, 3269, 5304, 5297), cub)]
    check("GEMM v2 share of cuBLAS, lowest", min(ratios), 0.41, "arithmetic", fmt="{:.2f}")
    check("GEMM v2 share of cuBLAS, highest", max(ratios), 0.49, "arithmetic", fmt="{:.2f}")
    lin = r"^\s+linear\s"
    check("original GEMM profiled ms", grab("results/final_profile.txt", lin, r"\s([\d.]+)\s+[\d.]+%"), 2425, "hw1-asr/results/final_profile.txt", tol=0.5, fmt="{:.0f}")
    check("original GEMM share %", grab("results/final_profile.txt", lin, r"([\d.]+)%"), 89, "hw1-asr/results/final_profile.txt", tol=0.5, fmt="{:.0f}")
    check("GEMM v2 profiled ms", grab("extensions/results/v2_e2e_dispatch.txt", lin, r"\s([\d.]+)\s+[\d.]+%"), 1271, "hw1-asr/extensions/results/v2_e2e_dispatch.txt", tol=0.5, fmt="{:.0f}")
    check("GEMM v2 share %", grab("extensions/results/v2_e2e_dispatch.txt", lin, r"([\d.]+)%"), 81, "hw1-asr/extensions/results/v2_e2e_dispatch.txt", tol=0.5, fmt="{:.0f}")
    text = "\n".join(log("extensions/results/v2_dispatch_racecheck.txt"))
    check("GEMM v2 Racecheck hazards", float(re.search(r"RACECHECK SUMMARY: (\d+) hazards", text).group(1)), 0,
          "hw1-asr/extensions/results/v2_dispatch_racecheck.txt", fmt="{:.0f}")


def asr_blocks():
    section("ASR: single-session sweep of GEMM v2's constants")
    dirs = sorted(glob.glob(str(ASR / "extensions" / "results" / "gemm_blocks_*")))
    if not dirs:
        MISSING.append("hw1-asr/extensions/results/gemm_blocks_*/")
        raise Missing("gemm_blocks_*")
    d = Path(dirs[-1]).relative_to(ASR)
    src = f"hw1-asr/{d}"
    shapes = (("Dec. q", r"decoder q_proj"), ("Dec. gate", r"decoder gate_proj"),
              ("Enc. q", r"encoder q_proj"), ("Enc. FC1", r"encoder fc1"))
    table = (("2x2, depth 16", "tm2_tn2_bn64_bk16", (1507, 1761, 2461, 2564), 2556.0),
             ("4x4, depth 8", "tm4_tn4_bn64_bk8", (1452, 2705, 4299, 4215), 1901.1),
             ("4x4, depth 16 (v2)", "tm4_tn4_bn64_bk16", (2019, 3333, 5329, 5048), 1593.2),
             ("4x4, depth 32", "tm4_tn4_bn64_bk32", (2152, 3747, 5747, 5522), 1481.9),
             ("8x4, depth 16", "tm8_tn4_bn64_bk16", (1192, 2893, 7396, 6418), 1824.1),
             ("8x8, depth 16", "tm8_tn8_bn64_bk16", (627, 1828, 6069, 6110), 2760.5))
    for name, tag, want, ms in table:
        for (col, pat), w in zip(shapes, want):
            check(f"{name}, {col}", grab(f"{d}/test_{tag}.txt", pat, GFLOPS), w, f"{src}/test_{tag}.txt", fmt="{:.0f}")
        check(f"{name}, ms per transcription", grab(f"{d}/e2e_{tag}.txt", "Time:", TIME), ms, f"{src}/e2e_{tag}.txt", fmt="{:.1f}")
    for tag in ("tm2_tn2_bn64_bk16", "tm4_tn4_bn64_bk8", "tm4_tn4_bn64_bk16", "tm4_tn4_bn64_bk32", "tm4_tn8_bn64_bk16",
                "tm8_tn4_bn64_bk16", "tm8_tn8_bn64_bk16", "tm8_tn8_bn128_bk16", "tm4_tn4_bn128_bk16"):
        lines = log(f"{d}/test_{tag}.txt")
        ok = sum(bool(re.search(r"PASS\s+max_err", x)) for x in lines)
        bad = sum(bool(re.search(r"FAIL\s+max_err", x)) for x in lines)
        check(f"{tag}: GEMM tests passed / failed", f"{ok} / {bad}", "10 / 0", f"{src}/test_{tag}.txt")
    check("PyTorch at the start of the job", grab(f"{d}/e2e_pytorch_start.txt", "Time:", TIME), 769.8, f"{src}/e2e_pytorch_start.txt", fmt="{:.1f}")
    check("PyTorch at the end of the job", grab(f"{d}/e2e_pytorch_end.txt", "Time:", TIME), 785.0, f"{src}/e2e_pytorch_end.txt", fmt="{:.1f}")
    check("drift over the job", 785.0 / 769.8 - 1, 0.02, "arithmetic", fmt="{:.2f}")
    check("depth 32 saving over v2", 1 - 1481.9 / 1593.2, 0.07, "arithmetic", fmt="{:.2f}")
    lines = log(f"{d}/test_tm8_tn4_bn64_bk16.txt")
    i = next(n for n, x in enumerate(lines) if re.search(r"encoder q_proj", x))
    cub = float(re.search(r"cuBLAS\s+(\d+) GFLOP/s", lines[i + 1]).group(1))
    check("8x4 encoder q_proj share of cuBLAS", 7396 / cub, 0.57, f"{src}/test_tm8_tn4_bn64_bk16.txt", fmt="{:.2f}")
    check("4x8 encoder q_proj", grab(f"{d}/test_tm4_tn8_bn64_bk16.txt", r"encoder q_proj", GFLOPS), 4839, f"{src}/test_tm4_tn8_bn64_bk16.txt", fmt="{:.0f}")
    check("128-column tile, ms per transcription", grab(f"{d}/e2e_tm4_tn4_bn128_bk16.txt", "Time:", TIME), 1859.2, f"{src}/e2e_tm4_tn4_bn128_bk16.txt", fmt="{:.1f}")


def derivations():
    section("Derived statements and appendix arithmetic")
    score = lambda dense_rank, bm25_rank: 0.95 / (60 + dense_rank) + 0.05 / (60 + bm25_rank)
    check("dense rank 7 with the best BM25 rank can beat rank 5 with the worst",
          "yes" if score(7, 1) > score(5, 100) else "no", "yes", "Equation 1, alpha 0.95")
    check("dense rank 8 with the best BM25 rank can beat rank 5 with the worst",
          "yes" if score(8, 1) > score(5, 100) else "no", "no", "Equation 1, alpha 0.95")
    check("encoder values rotated by RoPE, millions", 2 * 1500 * 20 * 32 / 1e6, 1.92, "arithmetic", fmt="{:.2f}")
    check("RoPE traffic each way, MB", 2 * 1500 * 20 * 32 * 4 / 1e6, 7.68, "arithmetic", fmt="{:.2f}")
    check("sine/cosine table, KB", 1500 * 16 * 2 * 4 / 1e3, 192, "arithmetic", fmt="{:.0f}")
    check("58-block partial MaxSum summaries, bytes", 58 * 2 * 4, 464, "arithmetic", fmt="{:.0f}")
    check("model-level RoPE calls", grab("results/final_profile.txt", r"^\s+rope\s", r"\s(\d+)\s+[\d.]+\s+[\d.]+%"),
          396, "hw1-asr/results/final_profile.txt", fmt="{:.0f}")
    check("ASR share of the latency budget: PyTorch ASR s", 768.9 / 1000, 0.77, "latency table", fmt="{:.2f}")
    check("median generation s at k=5", 4269 / 1000, 4.3, "Table 3", fmt="{:.1f}")


def main():
    print("Numbers quoted in the report, recomputed from saved files")
    print(f"Repository root: {ROOT.name}")
    for fn in (rag_dev, rag_rerank, rag_practice_tuning, rag_practice_minilm, rag_cost, rag_practice_depth,
               rag_followup_saved, asr_table, asr_rope_reduce, asr_gemm, asr_attention, asr_v2, asr_blocks, derivations):
        block(fn)
    print("\n=== SUMMARY ===")
    print(f"values checked: {COUNT[0]}")
    print(f"values that differ from the report: {len(DIFFS)}")
    for d in DIFFS:
        print(f"  DIFF {d}")
    print(f"result files not found: {len(MISSING)}")
    for m in MISSING:
        print(f"  MISSING {m}")
    sys.exit(1 if DIFFS or MISSING else 0)


if __name__ == "__main__":
    main()

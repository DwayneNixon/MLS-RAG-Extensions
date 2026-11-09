import json
import math
from statistics import mean, stdev, median, linear_regression

BASE = "rag/results/phase3"
CONFIGS = ["k2", "k5", "k10", "k20", "k5b"]

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

# Per query nomralised scores, one judge repeat each
scores = {}
for c in ["k2", "k10", "k20", "k5b"]:
    rows = load_jsonl(f"{BASE}/{c}/judge/answer_judge_per_query.jsonl")
    scores[c] = {r["query_id"]:r["normalised_answer_score"] for r in rows}

k5_repeats = {}
for r in load_jsonl(f"{BASE}/k5/judge/judge_cache.jsonl"):
    res = r["result"]
    s = (res["correctness"] + res["grounding"] +res["completeness"]) / 12
    k5_repeats.setdefault(r["query_id"], []).append(s)
scores["k5"] = {q: v[0] for q, v in k5_repeats.items()}

# Queries scored in every config
shared = set(scores["k2"])
for c in CONFIGS:
    shared &= set(scores[c])
print("shared queries:", len(shared))

# Paired differences against k5
for c in ["k5b", "k2", "k10", "k20"]:
    diffs = [scores[c][q] - scores["k5"][q] for q in shared]
    se_diff = stdev(diffs)/math.sqrt(len(diffs))
    print(f"{c} - k5: mean {mean(diffs):+.3f}  se {se_diff:.3f}")

# Judge noise: repeat 2 - repeat 1 on the same k5 answers
full = [q for q in shared if len(k5_repeats[q]) >= 2]
jd = [k5_repeats[q][1] - k5_repeats[q][0] for q in full]
print(f"judge rep2 - rep1: mean {mean(jd):+.3f}  se {stdev(jd)/math.sqrt(len(jd)):.3f}")

# Subgroups to check if k20 actually added relevant records over k5
ret = {r["query_id"]: r for r in load_jsonl("rag/results/retrieval_k_dev.jsonl")}

def gain(q):
    # relevant records in top 20 minus relevant records in top 5
    return round(20 * ret[q]["Precision@20"] - 5 * ret[q]["Precision@5"], 6)

def se(x):
    return stdev(x) / math.sqrt(len(x))

gained = [q for q in shared if gain(q) > 0]
none   = [q for q in shared if gain(q) == 0]
print("gained:", len(gained), " none:", len(none))

for name, group in [("gained", gained), ("none", none)]:
    d = [scores["k20"][q] - scores["k5"][q] for q in group]
    print(f"{name}: k20 - k5 mean {mean(d):+.3f}  se {se(d):.3f}")

# Length confound, completeness and answer length, by subgroup
def per_query(c, field):
    rows = load_jsonl(f"{BASE}/{c}/judge/answer_judge_per_query.jsonl")
    return {r["query_id"]: r[field] for r in rows}

# k5 repeat 1: first cache entry per query, kept whole
k5_first = {}
for r in load_jsonl(f"{BASE}/k5/judge/judge_cache.jsonl"):
    if r["query_id"] not in k5_first:
        k5_first[r["query_id"]] = r["result"]

comp = {"k20": per_query("k20", "completeness"),
        "k5":  {q: res["completeness"] for q, res in k5_first.items()}}

def tokens(c):
    return {r["query_id"]: r["answer_tokens"] for r in load_jsonl(f"{BASE}/{c}/answers.jsonl")}
tok = {"k20": tokens("k20"), "k5": tokens("k5")}

for name, group in [("gained", gained), ("none", none)]:
    dc = [comp["k20"][q] - comp["k5"][q] for q in group]     # completeness, k20 - k5 (0-4 scale)
    dt = [tok["k20"][q] - tok["k5"][q] for q in group]     # answer tokens, k20 - k5
    print(f"{name}: completeness {mean(dc):+.2f} (se {se(dc):.2f})  "
          f"tokens {mean(dt):+.0f} (se {se(dt):.0f})")

for c in ["k2", "k5", "k10", "k20"]:
    rows = load_jsonl(f"{BASE}/{c}/answers.jsonl")[1:]
    rows = [r for r in rows if r["query_id"] in shared]
    pt = [r["prompt_tokens"] for r in rows]
    at = [r["answer_tokens"] for r in rows]
    gm = [r["timing"]["generation_ms"] for r in rows]
    print(f"{c}: n {len(rows)}  prompt {median(pt):.0f}  answer {median(at):.0f}  gen_ms {median(gm):.0f}")

xs, ys = [], []
for c in ["k2", "k5", "k10", "k20"]:
    for r in load_jsonl(f"{BASE}/{c}/answers.jsonl")[1:]:
        xs.append(r["answer_tokens"])
        ys.append(r["timing"]["generation_ms"])
slope, intercept = linear_regression(xs, ys)
print(f"fit: generation_ms = {intercept:.0f} + {slope:.1f} * answer_tokens")

# Checking for floor effect: queries that score 0 in every k config
all_zero = [q for q in shared
            if all(scores[c][q] == 0  for c in ["k2", "k5", "k10", "k20"])]
print(f"zero in all k configs: {len(all_zero)} / {len(shared)}")

# Main comparison again, without the all-zero queries
rest = [q for q in shared if q not in all_zero]
for c in ["k2", "k10", "k20"]:
    d = [scores[c][q] - scores["k5"][q] for q in rest]
    print(f"{c} - k5 (non floor): mean {mean(d):+.3f}  se {se(d):.3f}")

# Reranker against k5 repeat 1, same queries, same method
# 9. Arm 2: rerankers against k5 repeat 1, same queries, same method
R = "rag/results"
for name, path in [("hybrid", f"{R}/hybrid_dev_judge_full125/answer_judge_per_query.jsonl"),
                   ("minilm", f"{R}/minilm_dev_judge_full125/answer_judge_per_query.jsonl")]:
    scores[name] = {r["query_id"]: r["normalised_answer_score"] for r in load_jsonl(path)}

shared9 = shared & set(scores["hybrid"]) & set(scores["minilm"])
print("shared with rerankers:", len(shared9))
for c in ["hybrid", "minilm"]:
    d = [scores[c][q] - scores["k5"][q] for q in shared9]
    print(f"{c} - k5: mean {mean(d):+.3f}  se {se(d):.3f}")

# MiniLM subgroups by change in Recall@5 (MiniLM top 5 vs dense top 5)
mret = {r["query_id"]: r for r in load_jsonl(f"{R}/minilm_dev_full125.jsonl")}
def rchange(q):
    return round(mret[q]["minilm_metrics"]["Recall@5"] - mret[q]["dense_metrics"]["Recall@5"], 6)

for label, test in [("lost", lambda x: x < 0), ("same", lambda x: x == 0), ("gained", lambda x: x > 0)]:
    group = [q for q in shared9 if test(rchange(q))]
    d = [scores["minilm"][q] - scores["k5"][q] for q in group]
    print(f"minilm {label} (n={len(group)}): mean {mean(d):+.3f}  se {se(d):.3f}")

# Generation cost for reranker
from collections import Counter
for name, path in [("hybrid", f"{R}/hybrid_dev_answers_full125.jsonl"),
                   ("minilm", f"{R}/minilm_dev_answers_full125.jsonl")]:
    rows = [r for r in load_jsonl(path) if r["query_id"] in shared]
    print(name, "answer_origin:", dict(Counter(str(r.get("answer_origin")) for r in rows)))
    groups = {"all": rows}
    for r in rows:
        groups.setdefault(str(r.get("answer_origin")), []).append(r)
    for label, sub in groups.items():
        pt = [r["prompt_tokens"] for r in sub if r.get("prompt_tokens") is not None]
        at = [r["answer_tokens"] for r in sub if r.get("answer_tokens") is not None]
        gm = [r["timing"]["generation_ms"] for r in sub if r.get("timing", {}).get("generation_ms") is not None]
        print(f"  {name} [{label}] n {len(sub)}  prompt {median(pt):.0f}  answer {median(at):.0f}  gen_ms {median(gm):.0f}")

# Redundancy test: score by number of relevant records sent to the generator
from collections import defaultdict
def n_rel(q, k):
    return round(k * ret[q][f"Precision@{k}"])   # relevant records in the top k

for c, k in [("k2", 2), ("k5", 5), ("k20", 20)]:
    groups = defaultdict(list)
    for q in shared:
        groups[min(n_rel(q, k), 3)].append(scores[c][q])
    for n in sorted(groups):
        g = groups[n]
        label = f"{n}+" if n == 3 else str(n)
        print(f"{c}: {label} relevant in top {k}: n={len(g):3d}  mean {mean(g):.3f}  se {se(g) if len(g) > 1 else 0:.3f}")
cross = [q for q in shared if n_rel(q, 5) == 0 and n_rel(q, 20) >= 1]
d = [scores["k20"][q] - scores["k5"][q] for q in cross]
print(f"0 -> 1+ relevant (k5 -> k20): n={len(cross)}  mean {mean(d):+.3f}  se {se(d):.3f}")

# Oracle evidence: gold records first or last, against k5 and against each other
P4 = "rag/results/phase4"
ids = {}
for c in ["gold_first", "gold_last"]:
    scores[c] = {r["query_id"]: r["normalised_answer_score"]
                 for r in load_jsonl(f"{P4}/{c}/judge/answer_judge_per_query.jsonl")}
    ids[c] = {r["query_id"]: r["retrieved_record_ids"] for r in load_jsonl(f"{P4}/{c}/answers.jsonl")}

s12 = shared & set(scores["gold_first"]) & set(scores["gold_last"])
print("oracle shared queries:", len(s12))
for c in ["gold_first", "gold_last"]:
    d = [scores[c][q] - scores["k5"][q] for q in s12]
    print(f"{c} - k5: mean {mean(d):+.3f}  se {se(d):.3f}")

# Position test: only queries where the two prompts actually differ in order
moved = [q for q in s12 if ids["gold_first"][q] != ids["gold_last"][q]]
d = [scores["gold_first"][q] - scores["gold_last"][q] for q in moved]
print(f"gold_first - gold_last (order differs, n={len(moved)}): mean {mean(d):+.3f}  se {se(d):.3f}")

# Where does the oracle help? Split by how many relevant records k5 already had
for label, test in [("k5 had 0 relevant", lambda n: n == 0), ("k5 had 1+ relevant", lambda n: n >= 1)]:
    g = [q for q in s12 if test(n_rel(q, 5))]
    d = [scores["gold_first"][q] - scores["k5"][q] for q in g]
    print(f"gold_first - k5, {label} (n={len(g)}): mean {mean(d):+.3f}  se {se(d):.3f}")

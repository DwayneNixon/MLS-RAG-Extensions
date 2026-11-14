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

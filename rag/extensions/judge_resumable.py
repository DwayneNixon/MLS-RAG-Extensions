"""Run rag/scripts/evaluate_answers.py to completion, applying a fixed exclusion rule.

The supplied judge script stops when one judgement cannot be obtained. Its
judge_cache.jsonl keeps every finished judgement, so re-running resumes. This
wrapper re-runs it and applies the rule fixed in ANALYSIS_PLAN_practice_depth.md:

  * transport errors (connection, timeout, HTTP 429/5xx) never exclude a query:
    wait and try again;
  * any other judge failure (unparseable or out-of-range output) after
    --api-retries requests, repeated on a second full attempt, excludes that
    query. Excluded IDs and the last error are written to excluded.jsonl.

The judging itself (prompt, model, temperature 0.0, scoring) is the unmodified
supplied script, called as a subprocess. Run from the repository root.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "evaluate_answers.py"
TRANSPORT = re.compile(r"connection|timed out|timeout|error code: (429|5\d\d)|rate.?limit|"
                       r"temporarily|unavailable|bad gateway", re.I)


def load_jsonl(path: Path):
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # truncated final cache line; the judge re-requests it
    return rows


def run_eval(candidates, split, out_dir, api_retries, allow_subset):
    cmd = [sys.executable, str(EVAL), "--candidates", str(candidates), "--split", split,
           "--out-dir", str(out_dir), "--repeats", "1", "--api-retries", str(api_retries)]
    if allow_subset:
        cmd.append("--allow-subset")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line)
        tail = tail[-40:]
    return proc.wait(), "".join(tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--api-retries", type=int, default=8)
    ap.add_argument("--max-excluded", type=int, default=5,
                    help="stop instead of excluding more than this many queries")
    ap.add_argument("--max-transport-waits", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_jsonl(args.candidates)
    if not all_rows:
        sys.exit(f"No candidate rows in {args.candidates}")
    excl_path = args.out_dir / "excluded.jsonl"
    excluded = {r["query_id"]: r for r in load_jsonl(excl_path)}
    filtered = args.out_dir / "candidates_judged.jsonl"
    failures, transport_waits = {}, 0

    while True:
        rows = [r for r in all_rows if r["query_id"] not in excluded]
        with filtered.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        code, tail = run_eval(filtered, args.split, args.out_dir, args.api_retries,
                              allow_subset=bool(excluded))
        if code == 0:
            print(f"[judge] finished: {len(rows)} judged, {len(excluded)} excluded "
                  f"{sorted(excluded)}", flush=True)
            return
        if "judge request failed" not in tail:
            sys.exit("[judge] evaluate_answers.py failed for a reason other than a judge "
                     "request; fix it and re-run (finished judgements are cached).")
        judged = {r["query_id"] for r in load_jsonl(args.out_dir / "judge_cache.jsonl")}
        failing = next((r["query_id"] for r in rows if r["query_id"] not in judged), None)
        if failing is None:
            sys.exit("[judge] could not identify the failing query; inspect the log.")
        message = tail.strip().splitlines()[-1][:500]
        if TRANSPORT.search(message):
            transport_waits += 1
            if transport_waits > args.max_transport_waits:
                sys.exit("[judge] the endpoint keeps failing; re-run later (progress is cached).")
            print(f"[judge] transport error on {failing}; waiting 120s "
                  f"({transport_waits}/{args.max_transport_waits})", flush=True)
            time.sleep(120)
            continue
        failures[failing] = failures.get(failing, 0) + 1
        print(f"[judge] {failing}: judge failure {failures[failing]}/2: {message}", flush=True)
        if failures[failing] < 2:
            time.sleep(30)
            continue
        if len(excluded) + 1 > args.max_excluded:
            sys.exit(f"[judge] would exclude more than {args.max_excluded} queries; stopping.")
        excluded[failing] = {"query_id": failing, "reason": message,
                             "api_attempts": 2 * args.api_retries}
        with excl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(excluded[failing], ensure_ascii=False) + "\n")
        print(f"[judge] excluded {failing} under the pre-specified rule", flush=True)


if __name__ == "__main__":
    main()

"""Summarise a GEMM block-size run: python extensions/summarise_gemm_blocks.py RESULTS_DIR

Reads the test_*.txt and e2e_*.txt logs written by run_gemm_blocks.sbatch and prints
one table: threads per block, tests passed, throughput on four model shapes with its
share of cuBLAS measured in the same test run, and whole-model transcription time.
"""
import re
import sys
from pathlib import Path

SHAPES = (("Dec q", r"decoder q_proj"), ("Dec gate", r"decoder gate_proj"),
          ("Enc q", r"encoder q_proj"), ("Enc FC1", r"encoder fc1"))


def read(path):
    return path.read_text(errors="replace").splitlines() if path.exists() else []


def throughput(lines):
    """{shape: (kernel GFLOP/s, cuBLAS GFLOP/s)} from one GEMM test log."""
    out = {}
    for label, pat in SHAPES:
        for i, line in enumerate(lines):
            if re.search(pat, line):
                mine = re.search(r"(\d+) GFLOP/s", line)
                ref = re.search(r"cuBLAS\s+(\d+) GFLOP/s", lines[i + 1]) if i + 1 < len(lines) else None
                if mine:
                    out[label] = (int(mine.group(1)), int(ref.group(1)) if ref else None)
                break
    return out


def passed(lines):
    ok = sum(bool(re.search(r"\bPASS\s+max_err", x)) for x in lines)
    bad = sum(bool(re.search(r"\bFAIL\s+max_err", x)) for x in lines)
    return ok, bad


def e2e(lines):
    for line in lines:
        m = re.search(r"Time:\s+([\d.]+) ms per transcription \(\+/- ([\d.]+) ms\)", line)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def linear_ms(lines):
    total = 0.0
    for line in lines:
        m = re.match(r"\s+linear\w*\s+\w+\s+\d+\s+([\d.]+)\s+[\d.]+%", line)
        if m:
            total += float(m.group(1))
    return total or None


def fmt_tp(tp, label):
    if label not in tp:
        return f"{'-':>12}"
    mine, ref = tp[label]
    return f"{mine:>6} {('(' + format(100 * mine / ref, '.0f') + '%)') if ref else '':>5}"


def main():
    d = Path(sys.argv[1])
    rows = []
    orig = read(d / "test_original.txt")
    if orig:
        rows.append(("assigned GEMM, tile 32", "1024 / 1024", orig, read(d / "e2e_original.txt")))
    for test in sorted(d.glob("test_tm*.txt")):
        m = re.match(r"test_tm(\d+)_tn(\d+)_bn(\d+)_bk(\d+)", test.stem)
        tm, tn, bn, bk = map(int, m.groups())
        name = f"{tm}x{tn} block, width {bn}, depth {bk}" + ("  <- GEMM v2" if (tm, tn, bn, bk) == (4, 4, 64, 16) else "")
        threads = f"{(64 // tm) * (bn // tn)} / {(32 // tm) * (bn // tn)}"
        rows.append((name, threads, read(test), read(d / f"e2e_{test.stem[5:]}.txt")))

    print("GEMM block-size experiment: throughput in GFLOP/s (share of cuBLAS in the same run)")
    print(f"{'configuration':<40} {'threads':>11} {'tests':>7} "
          + " ".join(f"{s:>12}" for s, _ in SHAPES) + f" {'GEMM ms':>8} {'ms/transcription':>18}")
    for name, threads, test, run in rows:
        ok, bad = passed(test)
        tp = throughput(test)
        t = e2e(run)
        lin = linear_ms(run)
        print(f"{name:<40} {threads:>11} {str(ok) + '/' + str(ok + bad):>7} "
              + " ".join(fmt_tp(tp, s) for s, _ in SHAPES)
              + f" {format(lin, '.0f') if lin else '-':>8}"
              + f" {(format(t[0], '.1f') + ' +/- ' + format(t[1], '.1f')) if t else 'not run':>18}")
    print("\nthreads: per block for 64-row tiles / 32-row tiles")
    for label, f in (("PyTorch at start", "e2e_pytorch_start.txt"), ("PyTorch at end", "e2e_pytorch_end.txt")):
        t = e2e(read(d / f))
        if t:
            print(f"{label}: {t[0]:.1f} +/- {t[1]:.1f} ms per transcription")
    lines = read(d / "test_original.txt")
    for i, line in enumerate(lines):
        if re.search(r"encoder q_proj", line) and i + 1 < len(lines):
            print("cuBLAS reference lines are the 'for comparison' rows under each case in test_*.txt")
            break


if __name__ == "__main__":
    main()

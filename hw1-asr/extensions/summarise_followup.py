"""Summarise a run_followup_gpu.sbatch output folder into tables (standard library only).

    python extensions/summarise_followup.py extensions/results/followup_kernels_JOBID
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1])


def text(name):
    p = OUT / f"{name}.txt"
    return p.read_text(errors="replace") if p.exists() else ""


def e2e(name):
    """ms per transcription, tokens, prompt tokens, GEMM share, transcript, status."""
    t = text(name)
    if not t:
        return None
    g = lambda pat, cast=str: (cast(m.group(1)) if (m := re.search(pat, t, re.M)) else None)
    gemm = sum(float(m.group(1)) for m in re.finditer(r"^\s+linear\w*\s+\w+\s+\d+\s+[\d.]+\s+([\d.]+)%", t, re.M))
    return {
        "ms": g(r"^Time: ([\d.]+) ms per transcription", float),
        "sd": g(r"\(\+/- ([\d.]+) ms\)", float),
        "tokens": g(r"run 1:\s+[\d.]+ ms\s+\((\d+) tokens\)", int),
        "prompt": g(r"prompt (\d+) tokens", int),
        "audio_s": g(r"^Audio: ([\d.]+) s", float),
        "gemm_share": gemm or None,
        "transcript": g(r"^Transcription: (.*)$"),
        "status": g(r"^Status: (\w+)"),
    }


def fmt(v, f="{:.1f}"):
    return "-" if v is None else f.format(v)


env = text("00_environment")
part = "clips" if "part: clips" in env else "kernels"
print(f"Follow-up summary ({part}), folder {OUT.name}")

start = e2e("01_pytorch_start") or e2e("02_pytorch_start")
end = e2e("08_pytorch_end") or e2e("04_pytorch_end")
if start and end:
    print(f"PyTorch at start {start['ms']:.1f} ms, at end {end['ms']:.1f} ms "
          f"(drift {100 * (end['ms'] / start['ms'] - 1):+.1f}%)")

if part == "kernels":
    print("\n== 2a. M sweep at N = K = 2048 (GFLOP/s; blocks / threads per block)")
    sweep = {}
    for p in sorted(OUT.glob("02_sweep_*.txt")):
        name = p.stem.replace("02_", "")
        rows = {}
        for m in re.finditer(r"^\s*(\d+)\s+2048\s+2048\s+[\d.]+\s+([\d.]+)\s+(\d+)\s+(\d+)", p.read_text(), re.M):
            rows[int(m.group(1))] = (float(m.group(2)), int(m.group(3)), int(m.group(4)))
        sweep[name] = rows
    Ms = sorted({m for r in sweep.values() for m in r})
    print(f"{'variant':18s}" + "".join(f"{m:>16d}" for m in Ms))
    for name, rows in sweep.items():
        print(f"{name:18s}" + "".join(f"{rows[m][0]:>7.0f} {rows[m][1]:>4d}/{rows[m][2]:<3d}" if m in rows else f"{'-':>16s}" for m in Ms))
    for bm in (32, 64):
        a, b = sweep.get(f"sweep_4x4_bm{bm}", {}), sweep.get(f"sweep_8x4_bm{bm}", {})
        cross = [m for m in Ms if m in a and m in b and b[m][0] >= a[m][0]]
        print(f"BM={bm}: smallest M where 8x4 >= 4x4: {cross[0] if cross else 'none in the sweep'}")

    print("\n== 2b. Profiler counters")
    occ = sorted(OUT.glob("03_ncu_*.txt")) + sorted(OUT.glob("06_ncu_*.txt"))
    if (OUT / "03_ncu_unavailable.txt").exists():
        print("ncu not found on this node")
    for p in occ:
        t = p.read_text(errors="replace")
        if "ERR_NVGPUCTRPERM" in t or "permission" in t.lower():
            print(f"{p.stem}: not permitted on this node (hardware counters blocked)")
            continue
        vals = re.findall(r"^\s+(\S+)\s+\S+\s+([\d.,]+)\s*$", t, re.M)
        print(f"{p.stem}: " + ("; ".join(f"{k} {v}" for k, v in vals) if vals else "no metric lines (see log)"))

    print("\n== 2c-d. v2 and v3 in one session")
    print(f"{'variant':15s} {'tests':>7s} {'racecheck':>22s} {'ms/transcription':>18s} {'GEMM share':>11s}  transcript")
    for name in ("v2", "v3", "v3_bk16", "v3_bk32", "v2_interleave", "v2_nopad"):
        t = text(f"04_test3_{name}")
        passed = len(re.findall(r"\s{2,}PASS\s+max_err", t))   # one line per test case
        failed = len(re.findall(r"\s{2,}FAIL\s+max_err", t))
        rc = []
        for sfx in ("M150", "M37"):
            r = text(f"05_racecheck_{name}_{sfx}")
            m = re.search(r"RACECHECK SUMMARY: (\d+) hazard", r) or re.search(r"ERROR SUMMARY: (\d+) error", r)
            rc.append(f"{sfx}:{m.group(1) if m else '-'}")
        e = e2e(f"07_e2e_{name}") or {}
        print(f"{name:15s} {passed:>3d}/{passed + failed:<3d} {' '.join(rc):>22s} "
              f"{fmt(e.get('ms')):>10s} +/- {fmt(e.get('sd')):<5s} {fmt(e.get('gemm_share')):>10s}%  "
              f"{e.get('transcript') or '-'} [{e.get('status') or '-'}]")
else:
    clips = json.loads((OUT / "clips.json").read_text()) if (OUT / "clips.json").exists() else []
    for c in clips:
        print(f"{c['clip']}: {c['seconds']} s, {len(c['questions'])} Practice questions {c['questions']}")
    print(f"\n{'clip':10s} {'backend':9s} {'audio s':>8s} {'tokens':>7s} {'max M':>6s} {'ms':>9s} "
          f"{'x PyTorch':>9s} {'GEMM %':>7s}  same text as PyTorch")
    for clip in ("reference", "clip_1q", "clip_3q", "clip_6q"):
        ref = e2e(f"03_{clip}_pytorch")
        for be in ("pytorch", "assigned", "v2", "v3"):
            e = e2e(f"03_{clip}_{be}")
            if not e:
                continue
            maxm = e["prompt"] + e["tokens"] - 1 if e["prompt"] and e["tokens"] else None
            ratio = e["ms"] / ref["ms"] if ref and e["ms"] and ref["ms"] else None
            same = "yes" if ref and e["transcript"] == ref["transcript"] else "NO"
            print(f"{clip:10s} {be:9s} {fmt(e['audio_s'], '{:.2f}'):>8s} {fmt(e['tokens'], '{:d}'):>7s} "
                  f"{fmt(maxm, '{:d}'):>6s} {fmt(e['ms']):>9s} {fmt(ratio, '{:.2f}'):>9s} "
                  f"{fmt(e['gemm_share']):>7s}  {same}")
    print("\nmax M = prompt tokens + generated tokens - 1: the decoder recomputes the whole")
    print("sequence each step (no key/value cache), so its last step multiplies this many rows.")

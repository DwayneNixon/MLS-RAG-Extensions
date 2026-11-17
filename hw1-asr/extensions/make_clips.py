"""Build the test clips for the multi-utterance ASR follow-up (standard library only).

Concatenates Practice-split question recordings from the RAG corpus into longer
16 kHz mono PCM16 WAV files, with 0.4 s of silence between questions, so the
decoder produces more tokens than the single 3.5 s reference clip (13 tokens).
Every clip stays under 29 s, inside the encoder's 30 s window.

    python extensions/make_clips.py ../rag/data/rag_corpus /tmp/clips_dir

Writes clip_1q.wav, clip_3q.wav, clip_6q.wav and clips.json (the source query IDs
and durations) into the output folder. Selection is fixed: Practice questions
sorted by query ID, shuffled with seed 0, taken in that order.
"""
import json
import random
import sys
import wave
from pathlib import Path

data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in (data_dir / "student_audio_manifest.jsonl").open() if l.strip()]
rows = sorted((r for r in rows if r.get("split") == "practice"), key=lambda r: r["query_id"])
random.Random(0).shuffle(rows)


def read(path):
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2 or f.getframerate() != 16000 or f.getnchannels() != 1:
            return None
        return f.readframes(f.getnframes())


gap = b"\x00\x00" * int(0.4 * 16000)
pool = iter(rows)
summary = []
for name, n in (("clip_1q", 1), ("clip_3q", 3), ("clip_6q", 6)):
    frames, used = b"", []
    while len(used) < n:
        r = next(pool, None)
        if r is None:
            break  # no more questions fit; keep the clip as it is
        pcm = read(data_dir / r["audio_file"])
        if pcm is None:
            continue
        candidate = frames + (gap if frames else b"") + pcm
        if len(candidate) / 2 / 16000 > 29.0:
            if used:
                break  # the next question would leave the 30 s window; stop this clip here
            continue
        frames = candidate
        used.append(r["query_id"])
    with wave.open(str(out_dir / f"{name}.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(frames)
    summary.append({"clip": name, "questions": used, "seconds": round(len(frames) / 2 / 16000, 2)})
    print(summary[-1])
(out_dir / "clips.json").write_text(json.dumps(summary, indent=2))

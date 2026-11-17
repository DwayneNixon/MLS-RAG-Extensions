"""End-to-end latency per stage for Practice audio queries, one ASR backend per call.

Uses the supplied pipeline unchanged (ASRAdapter -> enc_a -> exact dense top 5 ->
SimpleContextBuilder -> ELM). The ASR backend is chosen the supplied way, through
the environment the ASR worker inherits: ASR_CUDA=none (PyTorch), ASR_CUDA=all
(the assigned kernels), or ASR_CUDA=all with ASR_LIB pointing at a GEMM v2 build.
Model loading happens in prepare() and is excluded. The first query is a warm-up
and is excluded from the medians.

    python rag/extensions/e2e_latency.py --label pytorch --n 25 \
        --output rag/results/phase5/e2e_latency/pytorch.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.asr_adapter import ASRAdapter  # noqa: E402
from src.context_builder import SimpleContextBuilder  # noqa: E402
from src.data import CourseData  # noqa: E402
from src.encoder import LocalEncoder  # noqa: E402
from src.generator import ELMGenerator  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402
from src.retriever import ExactDenseRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="pytorch, assigned or v2 (recorded only)")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Results cannot be overwritten.")

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    items = sorted(data.audio_items("practice"), key=lambda r: r["query_id"])[: args.n + 1]
    asr = ASRAdapter()
    asr.prepare()  # loads the ASR model once; not timed
    pipe = RAGPipeline(
        asr=asr,
        encoder=LocalEncoder(ROOT / "models" / "enc_a"),
        retriever=ExactDenseRetriever(data_dir / "embeddings" / "corpus_enc_a.npy",
                                      data_dir / "embeddings" / "corpus_enc_a_ids.json"),
        data=data, context_builder=SimpleContextBuilder(), generator=ELMGenerator(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("x", encoding="utf-8") as f:
        for i, item in enumerate(items):
            res = pipe.answer_audio(data_dir / item["audio_file"], k=args.k)
            row = {"label": args.label, "query_id": item["query_id"], "warmup": i == 0,
                   "transcript": res["transcript"], "timing": res["timing"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(item["query_id"], {k: round(v) for k, v in res["timing"].items()}, flush=True)
    asr.close()
    timed = [r for r in rows if not r["warmup"]]
    print(f"\n{args.label}: medians over {len(timed)} queries (warm-up excluded)")
    for key in ("asr_ms", "embedding_ms", "retrieval_ms", "generation_ms", "end_to_end_ms"):
        print(f"  {key:15s} {median(r['timing'][key] for r in timed):9.1f}")


if __name__ == "__main__":
    main()

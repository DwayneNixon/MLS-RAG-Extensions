"""Generate answers from cached transcripts (no ASR), logging extra fields."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]     
sys.path.insert(0, str(ROOT))
import argparse
import json
import time
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.context_builder import SimpleContextBuilder
from src.generator import ELMGenerator

def load_cache(path: Path):
    cache = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["query_id"]] = row["transcript"]
    return cache

class ELMGeneratorWithUsage(ELMGenerator):
    """Same as ELMGenerator, but also returns token counts and finish reason."""

    def generate(self, prompt:str):
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Use the supplied evidence to answer the user's question. "
                        "Do not invent facts that are unsupported by the evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature = 0.1
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        usage = response.usage
        return {
            "answer": response.choices[0].message.content.strip(),
            "latency_ms": elapsed_ms,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "answer_tokens": usage.completion_tokens if usage else None,
            "finish_reason": response.choices[0].finish_reason,
        }


def select_evidence(retriever, qvec, k, mode, labels):                     
    """Return (record_ids sent to the generator, the retrieval result used)."""
    if mode == "dense":
        ret = retriever.retrieve(qvec, k=k)
        return ret["record_ids"], ret
    ret = retriever.retrieve(qvec, k=100)
    dense = ret["record_ids"]
    # gold = labelled relevant records, most relevant first, at most k
    gold = sorted([r for r, s in labels.items() if s > 0],
                  key=lambda r: labels[r], reverse=True)[:k]
    # fill the rest of the k slots from the dense ranking, skipping gold records
    fill = [r for r in dense if r not in gold][:k - len(gold)]
    # same records in both modes; only the order differs
    record_ids = gold + fill if mode == "gold_first" else fill + gold
    return record_ids, ret


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--config", required=True, help="e.g. baseline, k10, k20")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--evidence", choices=["dense", "gold_first", "gold_last"],   
                    default="dense", help="dense = normal retrieval; gold_* = oracle evidence")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.output.exists():
        ap.error("Results cannot be overwritten.")

    cache = load_cache(args.transcript_cache)

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    qrels = data.load_qrels(args.split) if args.evidence != "dense" else {}   

    missing = [item["query_id"] for item in items if item["query_id"] not in cache]
    if missing:
        sys.exit(f"Missing from cache: {len(missing)} queries, e.g. {missing[:3]}")

    if args.limit:
        items = items[:args.limit]

    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    generator = ELMGeneratorWithUsage()
    builder = SimpleContextBuilder()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for item in items:
            qid = item["query_id"]
            transcript = cache[qid]

            t0 = time.perf_counter()
            qvec = encoder.encode(transcript)
            embedding_ms = (time.perf_counter() - t0) * 1000.0

            record_ids, ret = select_evidence(retriever, qvec, args.k,          
                                              args.evidence, qrels.get(qid, {}))
            built = builder.build(transcript, record_ids, data)
            prompt_chars = len(built["prompt"])
            gen = generator.generate(built["prompt"])
            answer_chars = len(gen["answer"])


            row = {
                "query_id": qid,
                "split": args.split,
                "config": args.config,
                "k": args.k,
                "evidence": args.evidence,                                     
                "transcript": transcript,
                "answer": gen["answer"],
                "retrieved_record_ids": record_ids,                            
                "prompt_chars": prompt_chars,
                "prompt_tokens": gen["prompt_tokens"],
                "answer_chars": answer_chars,
                "answer_tokens": gen["answer_tokens"],
                "finish_reason": gen["finish_reason"],
                "timing": {
                    "embedding_ms": embedding_ms,
                    "retrieval_ms": ret["latency_ms"],
                    "generation_ms": gen["latency_ms"],
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(qid, args.config, gen["prompt_tokens"], gen["answer_tokens"], round(gen["latency_ms"]))

if __name__ == "__main__":
    main()
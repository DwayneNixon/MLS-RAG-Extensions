"""Pretrained neural reranking of an existing dense candidate set.

Model:
    cross-encoder/ms-marco-MiniLM-L6-v2

The original dense retriever supplies the candidate documents.
This class only reorders those candidates; it does not generate
new candidates or modify the University's RAG implementation.
"""

import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


class MiniLMReranker:
    def __init__(
        self,
        model_path,
        device="cuda",
        batch_size=16,
        max_length=512,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)

        if batch_size < 1 or max_length < 1:
            raise ValueError("Invalid batch size or max length")

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")

            gpu_name = torch.cuda.get_device_name(0)

            if "RTX 2080 Ti" not in gpu_name:
                raise RuntimeError(
                    f"Unexpected GPU for coursework: {gpu_name}"
                )

        start = time.perf_counter()

        self.model = CrossEncoder(
            str(self.model_path),
            device=device,
            max_length=max_length,
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        self.model_load_ms = (
            time.perf_counter() - start
        ) * 1000.0

    def rerank(
        self,
        query,
        candidate_ids,
        data,
        top_k=5,
    ):
        """Rerank supplied documents using neural relevance scores.

        Returns the complete ranking and selected top-k documents.
        Ties preserve original dense retrieval order.
        """

        candidate_ids = list(candidate_ids)

        if not candidate_ids:
            raise ValueError("Candidate list is empty")

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Duplicate candidate IDs")

        if top_k < 1 or top_k > len(candidate_ids):
            raise ValueError("Invalid top_k")

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()

        pairs = [
            (query, data.record_text(record_id))
            for record_id in candidate_ids
        ]

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        scores = np.asarray(
            raw_scores, dtype=np.float64
        ).reshape(-1)

        if len(scores) != len(candidate_ids):
            raise RuntimeError(
                "Cross-encoder returned the wrong number of scores"
            )

        if not np.isfinite(scores).all():
            raise RuntimeError(
                "Cross-encoder produced non-finite scores"
            )

        # Larger neural scores indicate higher relevance.
        # Equal scores retain the original dense order.
        order = sorted(
            range(len(candidate_ids)),
            key=lambda i: (-float(scores[i]), i),
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        ranked_ids = [
            candidate_ids[i] for i in order
        ]

        ranked_scores = [
            float(scores[i]) for i in order
        ]

        return {
            "record_ids": ranked_ids,
            "scores": ranked_scores,
            "top_record_ids": ranked_ids[:top_k],
            "rerank_ms": elapsed_ms,
            "candidate_count": len(candidate_ids),
        }

"""BM25 + dense rank fusion over an existing dense top-100 candidate set.

Uses only the supplied course corpus and Python standard library.
No external pretrained model weights are required.
"""

import math
import re
import time
from collections import Counter


TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text):
    """Deterministic, lowercase word tokenisation."""
    return TOKEN_PATTERN.findall(text.casefold())


class HybridReranker:
    def __init__(
        self,
        data,
        alpha=0.5,
        rrf_constant=60,
        k1=1.5,
        b=0.75,
    ):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if rrf_constant < 0 or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25/RRF parameters")

        self.data = data
        self.alpha = alpha
        self.rrf_constant = rrf_constant
        self.k1 = k1
        self.b = b

        start = time.perf_counter()

        self.doc_lengths = {}
        self.doc_freq = Counter()

        total_tokens = 0

        # Corpus-wide BM25 statistics: all 24,000 documents.
        for record_id, text in data.records.items():
            tokens = tokenize(text)

            self.doc_lengths[record_id] = len(tokens)
            self.doc_freq.update(set(tokens))
            total_tokens += len(tokens)

        self.num_docs = len(self.doc_lengths)
        if not self.num_docs or not total_tokens:
            raise ValueError("Corpus is empty")

        self.avg_doc_length = total_tokens / self.num_docs

        self.index_build_ms = (
            time.perf_counter() - start
        ) * 1000.0

    def idf(self, term):
        """Non-negative Robertson-style BM25 IDF."""
        df = self.doc_freq.get(term, 0)

        return math.log(
            1.0
            + (self.num_docs - df + 0.5) / (df + 0.5)
        )

    def bm25_score(self, query_terms, record_id):
        """Compute BM25 for one query/document pair."""
        tokens = tokenize(self.data.record_text(record_id))
        frequencies = Counter(tokens)

        doc_length = self.doc_lengths[record_id]

        norm = self.k1 * (
            1.0 - self.b
            + self.b * doc_length / self.avg_doc_length
        )

        score = 0.0

        for term in query_terms:
            frequency = frequencies.get(term, 0)

            if frequency == 0:
                continue

            score += self.idf(term) * (
                frequency * (self.k1 + 1.0)
                / (frequency + norm)
            )

        return score

    def rerank(self, query, candidate_ids, top_k=5):
        """Rerank dense candidates; preserve the full ordering."""
        start = time.perf_counter()

        candidate_ids = list(candidate_ids)

        if not candidate_ids:
            raise ValueError("Empty candidate list")

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Duplicate candidate IDs")

        if top_k < 1 or top_k > len(candidate_ids):
            raise ValueError("Invalid top_k")

        # Original list is already in descending dense-score order.
        dense_ranks = list(range(1, len(candidate_ids) + 1))

        query_terms = set(tokenize(query))

        bm25_scores = [
            self.bm25_score(query_terms, record_id)
            for record_id in candidate_ids
        ]

        # Deterministic tie-breaking: original dense rank.
        bm25_order = sorted(
            range(len(candidate_ids)),
            key=lambda i: (-bm25_scores[i], dense_ranks[i]),
        )

        bm25_ranks = [0] * len(candidate_ids)

        for rank, index in enumerate(bm25_order, start=1):
            bm25_ranks[index] = rank

        # Weighted reciprocal-rank fusion.
        fusion_scores = [
            (
                self.alpha
                / (self.rrf_constant + dense_ranks[i])
                + (1.0 - self.alpha)
                / (self.rrf_constant + bm25_ranks[i])
            )
            for i in range(len(candidate_ids))
        ]

        final_order = sorted(
            range(len(candidate_ids)),
            key=lambda i: (-fusion_scores[i], dense_ranks[i]),
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return {
            "record_ids": [
                candidate_ids[i] for i in final_order
            ],
            "scores": [
                fusion_scores[i] for i in final_order
            ],
            "bm25_scores": [
                bm25_scores[i] for i in final_order
            ],
            "bm25_ranks": [
                bm25_ranks[i] for i in final_order
            ],
            "dense_ranks": [
                dense_ranks[i] for i in final_order
            ],
            "top_record_ids": [
                candidate_ids[i]
                for i in final_order[:top_k]
            ],
            "rerank_ms": elapsed_ms,
            "candidate_count": len(candidate_ids),
        }

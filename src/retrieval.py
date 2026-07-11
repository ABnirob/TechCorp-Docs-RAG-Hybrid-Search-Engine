"""
Hybrid retrieval: fuses BM25 (sparse/keyword) and TF-IDF-or-embedding
(dense/semantic) search results into a single ranked list.

Why hybrid at all? Pure dense retrieval quietly fails on exact-match
queries (error codes, config keys, product names, acronyms like "SOC 2"
or "429") because embeddings smear these into their surrounding semantic
neighborhood. Pure sparse retrieval fails on paraphrases ("how do I stop
getting rate limited" vs. a doc that only says "429" and "Retry-After").
Combining both is the standard production pattern for exactly this reason.

Two fusion strategies are implemented:
  - Reciprocal Rank Fusion (RRF): combines by RANK, not raw score, so it
    is robust to BM25 and cosine-similarity living on completely different
    scales. This is the default and the one used in the eval harness.
  - Weighted score fusion: min-max normalizes both score lists then takes
    a weighted sum. Included for comparison in the eval notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.chunking import Chunk
from src.indexing import SparseIndex, DenseIndex


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    sparse_rank: int | None
    dense_rank: int | None


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def reciprocal_rank_fusion(
    sparse_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    k: int = 60,
) -> dict[int, float]:
    """Standard RRF: score(doc) = sum(1 / (k + rank)) across the rankers it appears in."""
    fused: dict[int, float] = {}
    for rank, (idx, _score) in enumerate(sparse_results):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _score) in enumerate(dense_results):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def weighted_fusion(
    sparse_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    sparse_weight: float,
    dense_weight: float,
) -> dict[int, float]:
    sparse_idx = [i for i, _ in sparse_results]
    sparse_scores = _minmax_normalize([s for _, s in sparse_results])
    dense_idx = [i for i, _ in dense_results]
    dense_scores = _minmax_normalize([s for _, s in dense_results])

    fused: dict[int, float] = {}
    for i, s in zip(sparse_idx, sparse_scores):
        fused[i] = fused.get(i, 0.0) + sparse_weight * s
    for i, s in zip(dense_idx, dense_scores):
        fused[i] = fused.get(i, 0.0) + dense_weight * s
    return fused


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], sparse_index: SparseIndex, dense_index: DenseIndex):
        self.chunks = chunks
        self.sparse_index = sparse_index
        self.dense_index = dense_index

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        fusion_strategy: str = None,
        candidate_pool: int = 20,
    ) -> list[RetrievedChunk]:
        top_k = top_k or config.TOP_K
        fusion_strategy = fusion_strategy or config.FUSION_STRATEGY

        sparse_results = self.sparse_index.search(query, top_k=candidate_pool)
        dense_results = self.dense_index.search(query, top_k=candidate_pool)

        sparse_ranks = {idx: r for r, (idx, _) in enumerate(sparse_results)}
        dense_ranks = {idx: r for r, (idx, _) in enumerate(dense_results)}

        if fusion_strategy == "rrf":
            fused = reciprocal_rank_fusion(sparse_results, dense_results, k=config.RRF_K)
        elif fusion_strategy == "weighted":
            fused = weighted_fusion(sparse_results, dense_results, config.SPARSE_WEIGHT, config.DENSE_WEIGHT)
        else:
            raise ValueError(f"Unknown fusion strategy '{fusion_strategy}'. Use 'rrf' or 'weighted'.")

        ranked_indices = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                score=fused[i],
                sparse_rank=sparse_ranks.get(i),
                dense_rank=dense_ranks.get(i),
            )
            for i in ranked_indices
        ]

    def retrieve_sparse_only(self, query: str, top_k: int = None) -> list[RetrievedChunk]:
        top_k = top_k or config.TOP_K
        results = self.sparse_index.search(query, top_k=top_k)
        return [RetrievedChunk(self.chunks[i], score, r, None) for r, (i, score) in enumerate(results)]

    def retrieve_dense_only(self, query: str, top_k: int = None) -> list[RetrievedChunk]:
        top_k = top_k or config.TOP_K
        results = self.dense_index.search(query, top_k=top_k)
        return [RetrievedChunk(self.chunks[i], score, None, r) for r, (i, score) in enumerate(results)]

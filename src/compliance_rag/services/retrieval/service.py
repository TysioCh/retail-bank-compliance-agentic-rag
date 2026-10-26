from __future__ import annotations

from typing import Protocol

from compliance_rag.core.retrieval import (
    dedupe_candidates,
    fuse_rankings,
    threshold_filter,
    top_k_by_score,
)
from compliance_rag.models import DocCandidate


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[DocCandidate]:
        ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[DocCandidate]) -> list[DocCandidate]:
        ...


class HybridRetrievalService:
    def __init__(
        self,
        bm25: Retriever,
        dense: Retriever,
        sparse: Retriever,
        reranker: Reranker,
        top_k_per_route: int = 50,
        rrf_k: int = 60,
        rrf_top_k: int = 50,
        reranker_top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.top_k_per_route = top_k_per_route
        self.rrf_k = rrf_k
        self.rrf_top_k = rrf_top_k
        self.reranker_top_k = reranker_top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[DocCandidate]:
        bm25_results = self.bm25.retrieve(query, self.top_k_per_route)
        dense_results = self.dense.retrieve(query, self.top_k_per_route)
        sparse_results = self.sparse.retrieve(query, self.top_k_per_route)

        merged = dedupe_candidates([*bm25_results, *dense_results, *sparse_results])
        merged_by_id = {candidate.chunk_id: candidate for candidate in merged}

        rankings = [
            [candidate.chunk_id for candidate in bm25_results],
            [candidate.chunk_id for candidate in dense_results],
            [candidate.chunk_id for candidate in sparse_results],
        ]
        fused_scores = fuse_rankings(rankings, k=self.rrf_k)
        top_ids = [item[0] for item in top_k_by_score(fused_scores, self.rrf_top_k)]
        coarse_candidates = [merged_by_id[item] for item in top_ids if item in merged_by_id]

        reranked = self.reranker.rerank(query, coarse_candidates)
        filtered = threshold_filter(reranked, self.score_threshold)
        return filtered[: self.reranker_top_k]

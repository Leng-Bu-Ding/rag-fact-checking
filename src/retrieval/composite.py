from __future__ import annotations

from collections.abc import Mapping

from src.retrieval.base import Retriever
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.types import RetrievalResult


class HybridRetriever:
    """Fuse multiple retrievers through deterministic reciprocal ranks."""

    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever requires at least two retrievers")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self._retrievers = dict(retrievers)
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        rankings = {
            name: retriever.search(query, top_k=self._candidate_k)
            for name, retriever in self._retrievers.items()
        }
        return reciprocal_rank_fusion(
            rankings,
            top_k=top_k,
            rrf_k=self._rrf_k,
        )

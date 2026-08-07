from __future__ import annotations

from typing import Protocol

from src.retrieval.dense import DenseIndex, TextEncoder
from src.retrieval.types import RetrievalResult


class Retriever(Protocol):
    """Common text-query interface shared by every retriever."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...


class DenseRetriever:
    """Adapt a vector index and text encoder to the common interface."""

    def __init__(
        self,
        index: DenseIndex,
        encoder: TextEncoder,
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self._index = index
        self._encoder = encoder
        self._batch_size = batch_size

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            return []
        vector = self._encoder.encode_queries(
            [query], batch_size=self._batch_size
        )[0]
        return self._index.search_vector(vector, top_k=top_k)

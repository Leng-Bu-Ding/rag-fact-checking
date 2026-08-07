from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from rank_bm25 import BM25Okapi

from src.data.chunking import DocumentChunk
from src.retrieval.types import RetrievalResult

Tokenizer = Callable[[str], list[str]]

_ENGLISH_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize_english(text: str) -> list[str]:
    """Tokenize English text deterministically for the sparse baseline."""
    return _ENGLISH_TOKEN_RE.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over metadata-preserving document chunks."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        tokenizer: Tokenizer = tokenize_english,
        include_title: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if not chunks:
            raise ValueError("BM25Index requires at least one chunk")
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        if epsilon < 0:
            raise ValueError("epsilon cannot be negative")

        self._chunks = tuple(chunks)
        self._tokenizer = tokenizer
        self._include_title = include_title
        corpus = [
            tokenizer(f"{chunk.title} {chunk.text}" if include_title else chunk.text)
            for chunk in self._chunks
        ]
        self._model = BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []

        scores = self._model.get_scores(query_tokens)
        ordered_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )
        limit = min(top_k, len(ordered_indices))
        return [
            RetrievalResult(
                score=float(scores[index]),
                rank=rank,
                chunk=self._chunks[index],
            )
            for rank, index in enumerate(ordered_indices[:limit], start=1)
        ]

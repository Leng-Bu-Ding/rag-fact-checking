from __future__ import annotations

from dataclasses import dataclass

from src.data.chunking import DocumentChunk


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked retrieval result with its score and full source chunk."""

    score: float
    rank: int
    chunk: DocumentChunk

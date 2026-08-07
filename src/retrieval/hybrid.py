from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.retrieval.types import RetrievalResult


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalResult]],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Fuse retriever rankings while retaining the original chunk provenance."""
    if not rankings:
        raise ValueError("at least one ranking is required")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    scores: dict[str, float] = {}
    chunks = {}
    for results in rankings.values():
        seen: set[str] = set()
        for result in results:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                raise ValueError("a ranking cannot contain duplicate chunks")
            seen.add(chunk_id)
            chunks[chunk_id] = result.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + result.rank
            )

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        RetrievalResult(
            score=scores[chunk_id],
            rank=rank,
            chunk=chunks[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered[:top_k], start=1)
    ]

def prioritize_title_mentions(
    question: str,
    results: Sequence[RetrievalResult],
    *,
    top_k: int,
) -> list[RetrievalResult]:
    """Promote evidence whose complete source title is named in the question."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    lowered = question.casefold()
    ordered = sorted(
        results,
        key=lambda result: (
            result.chunk.title.casefold() not in lowered,
            result.rank,
            result.chunk.chunk_id,
        ),
    )
    return [
        RetrievalResult(score=result.score, rank=rank, chunk=result.chunk)
        for rank, result in enumerate(ordered[:top_k], start=1)
    ]
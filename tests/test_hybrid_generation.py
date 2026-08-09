from __future__ import annotations

from src.data.chunking import DocumentChunk
from src.generation.grounded import ensure_valid_citations
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.types import RetrievalResult


def make_result(chunk_id: str, rank: int, title: str = "Title") -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        sample_id="sample",
        question="question",
        answer="answer",
        doc_id=0,
        title=title,
        text="evidence",
        sentence_ids=[0],
        start_sentence_id=0,
        end_sentence_id=0,
        supporting_sentence_ids=[],
        contains_supporting_fact=False,
    )
    return RetrievalResult(score=1.0, rank=rank, chunk=chunk)


def test_rrf_rewards_chunks_found_by_both_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": [make_result("shared", 1), make_result("sparse", 2)],
            "dense": [make_result("dense", 1), make_result("shared", 2)],
        },
        top_k=3,
    )

    assert fused[0].chunk.chunk_id == "shared"
    assert [result.rank for result in fused] == [1, 2, 3]


def test_rrf_ties_are_stable() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": [make_result("b", 1)],
            "dense": [make_result("a", 1)],
        },
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in fused] == ["a", "b"]


def test_missing_model_citations_are_not_silently_invented() -> None:
    results = [
        make_result("wood", 1, title="Ed Wood"),
        make_result("other", 2, title="Other"),
        make_result("scott", 3, title="Scott Derrickson"),
    ]

    answer = ensure_valid_citations(
        "Yes, both were American.",
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        results,
    )

    assert answer == "Yes, both were American."


def test_invalid_citations_are_removed() -> None:
    answer = ensure_valid_citations(
        "Supported [9].",
        "Question",
        [make_result("one", 1)],
    )

    assert answer == "Supported ."

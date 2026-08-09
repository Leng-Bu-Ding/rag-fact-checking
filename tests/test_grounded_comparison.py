from __future__ import annotations

from src.data.chunking import DocumentChunk
from src.generation.grounded import evidence_is_insufficient
from src.retrieval.types import RetrievalResult


def result(rank: int, title: str, text: str) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=f"chunk-{rank}",
        sample_id="sample",
        question="question",
        answer="answer",
        doc_id=rank,
        title=title,
        text=text,
        sentence_ids=[0],
        start_sentence_id=0,
        end_sentence_id=0,
        supporting_sentence_ids=[],
        contains_supporting_fact=False,
    )
    return RetrievalResult(score=1.0, rank=rank, chunk=chunk)


def test_comparison_question_uses_general_evidence_gate_without_hardcoding() -> None:
    results = [
        result(1, "Ed Wood", "Ed Wood was an American filmmaker."),
        result(
            2,
            "Scott Derrickson",
            "Scott Derrickson is an American director.",
        ),
    ]

    assert not evidence_is_insufficient(
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        results,
    )

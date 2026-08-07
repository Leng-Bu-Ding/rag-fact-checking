from __future__ import annotations

from src.data.chunking import DocumentChunk
from src.evaluation.answers import evaluate_answer, normalize_answer, token_f1
from src.generation.grounded import evidence_is_insufficient
from src.retrieval.types import RetrievalResult


def evidence(title: str, *, supporting: bool, rank: int) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=f"chunk-{rank}",
        sample_id="sample",
        question="Who directed the film?",
        answer="Jane Doe",
        doc_id=rank,
        title=title,
        text=f"{title} directed the film.",
        sentence_ids=[0],
        start_sentence_id=0,
        end_sentence_id=0,
        supporting_sentence_ids=[0] if supporting else [],
        contains_supporting_fact=supporting,
    )
    return RetrievalResult(score=1.0, rank=rank, chunk=chunk)


def test_hotpot_answer_normalization_and_f1() -> None:
    assert normalize_answer("The Jane Doe. [1]") == "jane doe"
    assert token_f1("Jane Doe directed it [1]", "Jane Doe") == 2 * 2 / 6


def test_answer_metrics_separate_correctness_and_citations() -> None:
    results = [evidence("Jane Doe", supporting=True, rank=1), evidence("Other", supporting=False, rank=2)]
    metrics = evaluate_answer(
        "Jane Doe [1] [2]",
        "Jane Doe",
        results,
        frozenset({("sample", "Jane Doe", 0)}),
    )

    assert metrics["exact_match"] == 1.0
    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_precision"] == 0.5
    assert metrics["citation_recall"] == 1.0


def test_clear_out_of_domain_question_abstains() -> None:
    results = [evidence("Jane Doe", supporting=False, rank=1)]

    assert evidence_is_insufficient("What is the zxqv orbital constant?", results)
    assert not evidence_is_insufficient("Who directed the film?", results)

from __future__ import annotations

import numpy as np
import pytest

from src.data.chunking import DocumentChunk
from src.data.financebench import finance_question_from_record
from src.pipelines.financebench import (
    build_finance_rankings,
    covered_questions,
    group_chunks_by_document,
)
from src.retrieval.dense import DenseIndex


def make_chunk(chunk_id: str, doc: str, page: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        sample_id=doc,
        question="",
        answer="",
        doc_id=0,
        title=doc,
        text=text,
        sentence_ids=[0],
        start_sentence_id=0,
        end_sentence_id=0,
        supporting_sentence_ids=[],
        contains_supporting_fact=False,
        page_number=page,
    )


def question(doc: str = "ACME_2023_10K"):
    return finance_question_from_record(
        {
            "financebench_id": "q1",
            "company": "ACME",
            "doc_name": doc,
            "question_type": "metrics-generated",
            "question": "What was revenue?",
            "answer": "$10 million",
            "justification": "",
            "doc_type": "10k",
            "doc_period": "2023",
            "doc_link": "https://example.com/a.pdf",
            "evidence": [
                {
                    "doc_name": doc,
                    "evidence_page_num": 0,
                    "evidence_text": "Revenue was $10 million.",
                }
            ],
        }
    )


def test_covered_questions_requires_real_document_chunks() -> None:
    chunks = [make_chunk("a", "ACME_2023_10K", 1, "revenue")]
    assert covered_questions([question(), question("MISSING")], chunks) == [question()]


def test_gold_or_answer_fields_in_finance_corpus_are_rejected() -> None:
    unsafe = make_chunk("a", "ACME_2023_10K", 1, "revenue")
    unsafe = DocumentChunk(**{**unsafe.to_dict(), "answer": "$10 million"})
    with pytest.raises(ValueError, match="evaluation-only"):
        group_chunks_by_document([unsafe])


def test_document_scope_never_returns_another_filing() -> None:
    chunks = [
        make_chunk("a", "ACME_2023_10K", 1, "revenue ten million"),
        make_chunk("b", "OTHER_2023_10K", 1, "revenue twenty million"),
    ]
    dense = DenseIndex(chunks, np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    rankings, _ = build_finance_rankings(
        chunks,
        [question()],
        dense,
        {"q1": np.asarray([0.0, 1.0], dtype=np.float32)},
        corpus_scope="document",
        ranking_depth=2,
        bm25_config={"include_title": True, "k1": 1.5, "b": 0.75, "epsilon": 0.25},
    )
    assert {item.chunk.title for item in rankings["dense"]["q1"]} == {"ACME_2023_10K"}

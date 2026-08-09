from __future__ import annotations

import json

import pytest

from src.data.chunking import DocumentChunk
from src.data.financebench import (
    FinanceBenchAdapter,
    apply_alternate_urls,
    finance_question_from_record,
    read_questions_jsonl,
    resolve_download_url,
    safe_pdf_filename,
    write_questions_jsonl,
)
from src.evaluation.financebench import evaluate_finance_query, summarize_finance_failures
from src.retrieval.types import RetrievalResult


def finance_record() -> dict:
    return {
        "financebench_id": "financebench_sample_001",
        "company": "ACME",
        "doc_name": "ACME_2023_10K",
        "question_type": "metrics-generated",
        "question": "What was revenue?",
        "answer": "$10 million",
        "justification": "The filing reports $10 million.",
        "doc_type": "10k",
        "doc_period": "2023",
        "doc_link": "https://example.com/acme.pdf",
        "evidence": [
            {
                "doc_name": "ACME_2023_10K",
                "evidence_page_num": 7,
                "evidence_text": "Revenue was $10 million.",
                "evidence_text_full_page": "must never become index text",
            }
        ],
    }


def chunk(name: str, page: int, chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        sample_id=name,
        question="",
        answer="",
        doc_id=0,
        title=name,
        text="ordinary PDF text",
        sentence_ids=[0],
        start_sentence_id=0,
        end_sentence_id=0,
        supporting_sentence_ids=[],
        contains_supporting_fact=False,
        page_number=page,
    )


def test_adapter_builds_unique_document_manifest_without_gold_text() -> None:
    adapter = FinanceBenchAdapter.from_records([finance_record()])
    documents = adapter.documents()
    assert len(documents) == 1
    assert documents[0].local_path.endswith("ACME_2023_10K.pdf")
    assert "evidence" not in json.dumps(documents[0].to_dict()).casefold()
    assert "must never" not in json.dumps(documents[0].to_dict()).casefold()


def test_question_jsonl_round_trip_converts_source_index_to_pdf_page(tmp_path) -> None:
    question = finance_question_from_record(finance_record())
    output = tmp_path / "questions.jsonl"
    assert write_questions_jsonl([question], output) == 1
    loaded = read_questions_jsonl(output)
    assert loaded == [question]
    assert loaded[0].evidence[0].source_page_index == 7
    assert loaded[0].evidence[0].page_number == 8


def test_negative_source_page_index_is_rejected() -> None:
    record = finance_record()
    record["evidence"][0]["evidence_page_num"] = -1
    with pytest.raises(ValueError, match="cannot be negative"):
        finance_question_from_record(record)


def test_finance_metrics_distinguish_document_and_evidence_page_hits() -> None:
    question = finance_question_from_record(finance_record())
    results = [
        RetrievalResult(2.0, 1, chunk("ACME_2023_10K", 3, "wrong-page")),
        RetrievalResult(1.0, 2, chunk("ACME_2023_10K", 8, "gold-page")),
    ]
    metrics = evaluate_finance_query(results, question, ks=(1, 2))
    assert metrics["document_hit_at_1"] == 1.0
    assert metrics["evidence_page_hit_at_1"] == 0.0
    assert metrics["evidence_page_recall_at_2"] == 1.0
    assert metrics["mrr"] == 0.5


def test_summarize_finance_failures_separates_document_and_page_failures() -> None:
    evaluation = {
        "questions": [
            {
                "question_type": "novel-generated",
                "metrics": {
                    "document_hit_at_10": 1.0,
                    "evidence_page_hit_at_10": 1.0,
                },
            },
            {
                "question_type": "novel-generated",
                "metrics": {
                    "document_hit_at_10": 1.0,
                    "evidence_page_hit_at_10": 0.0,
                },
            },
            {
                "question_type": "metrics-generated",
                "metrics": {
                    "document_hit_at_10": 0.0,
                    "evidence_page_hit_at_10": 0.0,
                },
            },
        ]
    }

    summary = summarize_finance_failures(evaluation, k=10)

    assert summary["outcomes"]["gold_page_retrieved"]["count"] == 1
    assert summary["outcomes"]["correct_document_wrong_page"]["count"] == 1
    assert summary["outcomes"]["correct_document_not_retrieved"]["count"] == 1
    assert summary["by_question_type"]["novel-generated"]["question_count"] == 2


def test_safe_pdf_filename_is_deterministic() -> None:
    assert safe_pdf_filename("ACME 2023 / 10-K") == "ACME_2023_10-K.pdf"


def test_adobe_pdf_viewer_url_is_unwrapped() -> None:
    encoded = "aHR0cHM6Ly9leGFtcGxlLmNvbS9maWxpbmcucGRm"
    source = f"https://www.adobe.com/pdf-page.html?pdfTarget={encoded}"
    assert resolve_download_url(source) == "https://example.com/filing.pdf"


def test_alternate_url_preserves_original_provenance() -> None:
    document = FinanceBenchAdapter.from_records([finance_record()]).documents()[0]
    updated = apply_alternate_urls(
        [document], {document.doc_name: "https://mirror.example.com/acme.pdf"}
    )[0]
    assert updated.source_url == "https://mirror.example.com/acme.pdf"
    assert updated.original_source_url == document.source_url
    assert updated.status == "failed"

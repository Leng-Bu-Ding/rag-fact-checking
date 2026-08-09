from __future__ import annotations

import json

import pytest

from src.data.chunking import DocumentChunk
from src.generation.calculator import calculate
from src.generation.finance import OpenAICompatibleFinanceGenerator
from src.retrieval.types import RetrievalResult
from scripts.run_financebench_generation_evaluation import numeric_match


def result() -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id="c1",
        sample_id="ACME",
        question="",
        answer="",
        doc_id=1,
        title="ACME_2023_10K",
        text="Revenue increased from 80 to 100 million dollars.",
        sentence_ids=[0],
        start_sentence_id=0,
        end_sentence_id=0,
        supporting_sentence_ids=[],
        contains_supporting_fact=False,
        page_number=7,
    )
    return RetrievalResult(score=1.0, rank=1, chunk=chunk)


def test_calculator_allows_arithmetic_without_python_execution() -> None:
    assert calculate("(100 - 80) / 80 * 100") == 25.0
    with pytest.raises(ValueError, match="unsupported"):
        calculate("__import__('os').system('whoami')")


def test_generator_uses_calculator_then_returns_only_valid_citations() -> None:
    responses = iter(
        [
            {
                "choices": [{"message": {"content": json.dumps({
                    "answer": "",
                    "citations": [1],
                    "needs_calculation": True,
                    "calculation_expression": "(100 - 80) / 80 * 100",
                })}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            {
                "choices": [{"message": {"content": json.dumps({
                    "answer": "Revenue increased by 25%.",
                    "citations": [1, 9],
                })}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6},
            },
        ]
    )

    def fake_post(url, headers, payload, timeout):
        assert "response_format" in payload
        return next(responses)

    generator = OpenAICompatibleFinanceGenerator(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="test-model",
        post_json=fake_post,
    )
    generated = generator.generate("By what percent did revenue increase?", [result()])
    assert generated.answer == "Revenue increased by 25%."
    assert generated.calculation_result == 25.0
    assert generated.citations == (1,)
    assert generated.invalid_citations == (9,)
    assert generated.request_count == 2
    assert generated.prompt_tokens == 22


def test_numeric_match_handles_commas_and_small_rounding_error() -> None:
    assert numeric_match("The result was $1,000.50.", "$1,000.5") == 1.0
    assert numeric_match("The result was 90.", "100") == 0.0

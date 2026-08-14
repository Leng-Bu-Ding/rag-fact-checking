from __future__ import annotations

import json
import os

import pytest
import requests

from src.data.chunking import DocumentChunk
from src.generation.calculator import calculate
from src.generation.finance import (
    OpenAICompatibleFinanceGenerator,
    _default_post,
    _safe_error_text,
    analysis_plan_issues,
    probe_openai_compatible_api,
    validate_api_credentials,
)
from src.retrieval.types import RetrievalResult
from scripts.run_financebench_generation_evaluation import (
    generation_output_path,
    load_local_environment,
    numeric_match,
    rescore_record,
    select_question_records,
)
from src.generation.finance import financial_guidance


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
    with pytest.raises(ValueError, match="unsupported"):
        calculate("25 % 4")


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
        assert payload["enable_thinking"] is False
        return next(responses)

    generator = OpenAICompatibleFinanceGenerator(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="test-model",
        request_options={"enable_thinking": False},
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
    assert numeric_match("Cash dropped by 41.68% in FY2024.", "a decline of ~42%") == 1.0
    assert numeric_match("Cash increased by 38.0%.", "a decline of ~42%") == 0.0


def test_api_error_exposes_provider_details_without_headers(monkeypatch) -> None:
    class ErrorResponse:
        ok = False
        status_code = 400
        text = '{"code":"InvalidParameter","message":"Json mode is not supported"}'
        headers = {"content-type": "application/json"}

        @staticmethod
        def json():
            return {
                "code": "InvalidParameter",
                "message": "Json mode is not supported while thinking is enabled",
                "request_id": "request-123",
            }

    monkeypatch.setattr("src.generation.finance.requests.post", lambda *args, **kwargs: ErrorResponse())
    with pytest.raises(RuntimeError) as captured:
        _default_post(
            "https://api.example.com/chat/completions",
            {"Authorization": "Bearer secret-value"},
            {"model": "test-model"},
            90,
        )
    message = str(captured.value)
    assert "InvalidParameter" in message
    assert "request-123" in message
    assert "secret-value" not in message


def test_plain_text_api_error_is_visible_and_secrets_are_redacted(monkeypatch) -> None:
    class ErrorResponse:
        ok = False
        status_code = 400
        text = "invalid model for key sk-secret-value"
        headers = {"content-type": "text/plain", "x-request-id": "request-456"}

        @staticmethod
        def json():
            raise requests.exceptions.JSONDecodeError("invalid", "", 0)

    monkeypatch.setattr("src.generation.finance.requests.post", lambda *args, **kwargs: ErrorResponse())
    with pytest.raises(RuntimeError) as captured:
        _default_post(
            "https://api.example.com/chat/completions",
            {"Authorization": "Bearer secret-value"},
            {"model": "test-model"},
            90,
        )
    message = str(captured.value)
    assert "invalid model" in message
    assert "request-456" in message
    assert "sk-secret-value" not in message
    assert "[REDACTED]" in message


def test_safe_error_text_limits_and_compacts_output() -> None:
    assert _safe_error_text("  bad\n request  ") == "bad request"
    assert len(_safe_error_text("x" * 2000)) == 1000
    assert _safe_error_text("failed for sk-part.one_two-three") == "failed for [REDACTED]"


def test_default_transport_can_disable_environment_proxies(monkeypatch) -> None:
    observed = {}

    class SuccessResponse:
        ok = True

        @staticmethod
        def json():
            return {"choices": []}

    class FakeSession:
        trust_env = True

        def post(self, url, headers, json, timeout):
            observed["trust_env"] = self.trust_env
            return SuccessResponse()

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr("src.generation.finance.requests.Session", FakeSession)
    response = _default_post(
        "https://api.example.com/chat/completions",
        {"Authorization": "Bearer secret-value"},
        {"model": "test-model"},
        90,
        trust_env_proxy=False,
    )
    assert response == {"choices": []}
    assert observed == {"trust_env": False, "closed": True}


def test_api_credentials_reject_hidden_whitespace() -> None:
    with pytest.raises(ValueError, match="API Key contains"):
        validate_api_credentials("https://api.example.com/v1", "sk-test\r\n", "test-model")


def test_api_probe_stops_at_first_failure_without_exposing_key(monkeypatch) -> None:
    payloads = []

    class SuccessResponse:
        ok = True

        @staticmethod
        def json():
            return {"id": "request-1", "choices": [{"message": {"content": "OK"}}]}

    def fake_post(url, headers, json, timeout):
        payloads.append(json)
        if len(payloads) == 2:
            raise RuntimeError("HTTP 400 for sk-private-value")
        return SuccessResponse()

    class FakeSession:
        trust_env = True

        def post(self, url, headers, json, timeout):
            return fake_post(url, headers, json, timeout)

        def close(self):
            return None

    monkeypatch.setattr("src.generation.finance.requests.Session", FakeSession)
    results = probe_openai_compatible_api(
        base_url="https://api.example.com/v1",
        api_key="sk-private-value",
        model="test-model",
        trust_env_proxy=False,
    )
    assert [item.name for item in results] == ["minimal_chat", "non_thinking_chat"]
    assert results[0].ok is True
    assert results[1].ok is False
    assert "sk-private-value" not in results[1].detail
    assert "[REDACTED]" in results[1].detail


def test_local_environment_loads_without_overriding_terminal(tmp_path, monkeypatch) -> None:
    local_env = tmp_path / ".env.local"
    local_env.write_text(
        "# private\nRAG_API_KEY=sk-file.value\nRAG_API_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setenv("RAG_API_MODEL", "terminal-model")
    load_local_environment(local_env)
    assert os.environ["RAG_API_KEY"] == "sk-file.value"
    assert os.environ["RAG_API_MODEL"] == "terminal-model"


def test_financial_guidance_distinguishes_working_capital_from_current_ratio() -> None:
    working_capital = financial_guidance("Does the company have positive working capital?")
    current_ratio = financial_guidance("What is the working capital ratio?")
    assert [item.guidance_id for item in working_capital] == ["operating_working_capital"]
    assert [item.guidance_id for item in current_ratio] == ["current_ratio"]


def test_evaluation_sets_are_ordered_and_versioned() -> None:
    records = [{"financebench_id": "b"}, {"financebench_id": "a"}]
    selected = select_question_records(records, "dev", {"dev": ["a", "b"]})
    assert [item["financebench_id"] for item in selected] == ["a", "b"]
    path = generation_output_path(
        {"predictions_path_template": "results/{evaluation_set}_v2.jsonl"},
        "predictions",
        "holdout",
    )
    assert path.name == "holdout_v2.jsonl"


def test_analysis_plan_rejects_aggregate_operating_working_capital() -> None:
    issues = analysis_plan_issues(
        "Does the company have positive working capital?",
        {
            "needs_calculation": True,
            "selected_items": [
                {"label": "Total current assets", "value": 100},
                {"label": "Total current liabilities", "value": 80},
            ],
        },
    )
    assert issues and "Operating working capital" in issues[0]


def test_analysis_plan_requires_every_visible_working_capital_line_item() -> None:
    issues = analysis_plan_issues(
        "Does the company have positive working capital?",
        {
            "needs_calculation": True,
            "selected_items": [
                {"label": "Cash and cash equivalents", "role": "exclude", "value": 10},
                {"label": "Trade accounts receivable", "role": "add", "value": 20},
                {"label": "Accounts payable", "role": "subtract", "value": 5},
            ],
        },
        [{"text": "Cash and cash equivalents 10 Trade accounts receivable 20 "
                  "Other current assets 7 Accounts payable 5"}],
    )
    assert issues and "other current assets" in issues[0]


def test_analysis_plan_requires_quantified_period_comparison() -> None:
    issues = analysis_plan_issues(
        "Was there any drop in cash between FY2023 and FY2024?",
        {
            "needs_calculation": False,
            "selected_items": [
                {"label": "Cash FY2023", "value": 100},
                {"label": "Cash FY2024", "value": 80},
            ],
        },
    )
    assert issues and "Quantify" in issues[0]


def test_analysis_plan_requires_relative_period_change() -> None:
    issues = analysis_plan_issues(
        "Was there any drop in cash between FY2023 and FY2024?",
        {
            "needs_calculation": True,
            "calculation_expression": "100 - 80",
            "selected_items": [
                {"label": "Cash FY2023", "value": 100},
                {"label": "Cash FY2024", "value": 80},
            ],
        },
    )
    assert issues and "relative percentage" in issues[0]


def test_cached_generation_records_are_rescored() -> None:
    record = {
        "reference": "a decline of ~42%",
        "prediction": {"answer": "Cash dropped by 41.68% in FY2024."},
        "metrics": {"numeric_match": 0.0},
    }
    rescore_record(record)
    assert record["metrics"]["numeric_match"] == 1.0

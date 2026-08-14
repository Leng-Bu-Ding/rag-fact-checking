from __future__ import annotations

import json

import pytest

from src.evaluation.finance_answers import OpenAICompatibleFinanceAnswerJudge


def test_answer_judge_validates_and_normalizes_response() -> None:
    def fake_post(url, headers, payload, timeout):
        assert payload["response_format"] == {"type": "json_object"}
        assert "Reference answer" in payload["messages"][1]["content"]
        return {
            "choices": [{"message": {"content": json.dumps({
                "verdict": "Correct",
                "error_type": "wrong_value",
                "reason": "Equivalent after rounding.",
            })}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }

    judge = OpenAICompatibleFinanceAnswerJudge(
        base_url="https://api.example.com/v1",
        api_key="sk-test.value",
        model="test-model",
        post_json=fake_post,
    )
    result = judge.judge("What is revenue?", "$100 million", "$100.0 million")
    assert result.verdict == "correct"
    assert result.error_type == "none"
    assert result.prompt_tokens == 50


def test_answer_judge_rejects_unknown_verdict() -> None:
    def fake_post(url, headers, payload, timeout):
        return {"choices": [{"message": {"content": '{"verdict":"maybe","reason":"x"}'}}]}

    judge = OpenAICompatibleFinanceAnswerJudge(
        base_url="https://api.example.com/v1",
        api_key="sk-test.value",
        model="test-model",
        post_json=fake_post,
    )
    with pytest.raises(ValueError, match="invalid answer judge verdict"):
        judge.judge("q", "r", "p")

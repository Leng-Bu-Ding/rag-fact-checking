from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import requests

from src.generation.calculator import calculate
from src.retrieval.types import RetrievalResult

PostJson = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]


@dataclass(frozen=True)
class FinanceGeneration:
    answer: str
    citations: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    calculation_expression: str | None
    calculation_result: float | None
    model: str
    provider: str
    prompt_tokens: int | None
    completion_tokens: int | None
    request_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evidence_payload(results: Sequence[RetrievalResult], max_evidence: int) -> list[dict[str, Any]]:
    return [
        {
            "citation": index,
            "document": item.chunk.title,
            "page": item.chunk.page_number,
            "text": item.chunk.text,
        }
        for index, item in enumerate(results[:max_evidence], start=1)
    ]


def build_analysis_prompt(question: str, evidence: Sequence[dict[str, Any]]) -> str:
    return (
        "Answer this financial question using only the supplied PDF evidence. "
        "Return JSON with keys answer, citations, needs_calculation, "
        "calculation_expression. Citations must be integer evidence IDs. If "
        "arithmetic is required, set needs_calculation=true and provide an "
        "expression containing numbers and + - * / % ** only. Do not calculate "
        "mentally. If evidence is insufficient, say so and use no citations.\n\n"
        f"Question: {question}\nEvidence:\n"
        f"{json.dumps(list(evidence), ensure_ascii=False)}"
    )


def build_calculated_prompt(
    question: str,
    evidence: Sequence[dict[str, Any]],
    expression: str,
    result: float,
) -> str:
    return (
        "Produce the final concise answer using only the evidence and verified "
        "calculator output. Return JSON with keys answer and citations. Do not "
        "invent units or facts.\n\n"
        f"Question: {question}\nCalculator: {expression} = {result}\nEvidence:\n"
        f"{json.dumps(list(evidence), ensure_ascii=False)}"
    )


def _default_post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return dict(response.json())


def _parse_json_content(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if value.startswith("json"):
            value = value[4:].lstrip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("generator response must be a JSON object")
    return parsed


class OpenAICompatibleFinanceGenerator:
    """Evidence-only generator with an explicit, sandboxed calculator round trip."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 90,
        provider: str = "openai-compatible",
        post_json: PostJson = _default_post,
    ) -> None:
        if not base_url.strip() or not api_key.strip() or not model.strip():
            raise ValueError("base_url, api_key, and model are required")
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._timeout = timeout
        self._provider = provider
        self._post_json = post_json

    @property
    def metadata(self) -> dict[str, Any]:
        return {"provider": self._provider, "model": self._model, "endpoint": self._url}

    def _request(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self._post_json(
            self._url,
            self._headers,
            {
                "model": self._model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a grounded financial QA system."},
                    {"role": "user", "content": prompt},
                ],
            },
            self._timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("invalid chat completion response") from error
        return _parse_json_content(str(content)), dict(response.get("usage", {}))

    def generate(
        self,
        question: str,
        results: Sequence[RetrievalResult],
        *,
        max_evidence: int = 5,
    ) -> FinanceGeneration:
        if not question.strip():
            raise ValueError("question cannot be empty")
        evidence = evidence_payload(results, max_evidence)
        first, usage = self._request(build_analysis_prompt(question, evidence))
        expression = None
        calculation_result = None
        request_count = 1
        final = first
        if bool(first.get("needs_calculation")):
            expression = str(first.get("calculation_expression", "")).strip()
            calculation_result = calculate(expression)
            final, second_usage = self._request(
                build_calculated_prompt(question, evidence, expression, calculation_result)
            )
            usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0))
                + int(second_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0))
                + int(second_usage.get("completion_tokens", 0)),
            }
            request_count = 2
        answer = str(final.get("answer", "")).strip()
        if not answer:
            raise ValueError("generator returned an empty answer")
        raw_citations = tuple(int(value) for value in final.get("citations", []))
        valid = tuple(sorted({value for value in raw_citations if 1 <= value <= len(evidence)}))
        invalid = tuple(sorted({value for value in raw_citations if value not in valid}))
        return FinanceGeneration(
            answer=answer,
            citations=valid,
            invalid_citations=invalid,
            calculation_expression=expression,
            calculation_result=calculation_result,
            model=self._model,
            provider=self._provider,
            prompt_tokens=(int(usage["prompt_tokens"]) if "prompt_tokens" in usage else None),
            completion_tokens=(
                int(usage["completion_tokens"]) if "completion_tokens" in usage else None
            ),
            request_count=request_count,
        )

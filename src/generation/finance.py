from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import requests

from src.generation.calculator import calculate
from src.retrieval.types import RetrievalResult

PostJson = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]

_SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9._-]+")
PROMPT_VERSION = "financebench-v5"


@dataclass(frozen=True)
class MetricGuidance:
    guidance_id: str
    trigger: re.Pattern[str]
    instruction: str


_METRIC_GUIDANCE = (
    MetricGuidance(
        "operating_working_capital",
        re.compile(r"\bworking capital\b(?!\s+ratio)", re.IGNORECASE),
        "For analytical working capital, use net operating working capital: non-cash "
        "current operating assets minus non-debt current operating liabilities. Exclude "
        "cash, cash equivalents, short-term investments, short-term borrowings and the "
        "current portion of interest-bearing debt. Audit every individually listed current "
        "asset and current liability; include excluded items in selected_items with role "
        "exclude so coverage can be validated.",
    ),
    MetricGuidance(
        "current_ratio",
        re.compile(r"\b(?:working capital|current) ratio\b", re.IGNORECASE),
        "Current (working-capital) ratio = total current assets / total current liabilities, "
        "unless the question explicitly supplies another definition.",
    ),
    MetricGuidance(
        "quick_ratio",
        re.compile(r"\bquick ratio\b", re.IGNORECASE),
        "Quick ratio = liquid current assets / current liabilities. Prefer explicitly listed "
        "cash and equivalents, short-term investments and net receivables; exclude inventory "
        "and prepaid or other non-quick current assets. State the included line items.",
    ),
    MetricGuidance(
        "capital_intensity",
        re.compile(r"\bcapital[- ]intens(?:ive|ity)\b", re.IGNORECASE),
        "Capital intensity may be assessed with total assets / revenue. When available, "
        "cross-check with capex / revenue, fixed assets / total assets and return on assets; "
        "do not claim a universal threshold and explain the judgment.",
    ),
    MetricGuidance(
        "return_on_assets",
        re.compile(r"\b(?:return on assets|ROA)\b", re.IGNORECASE),
        "Follow any formula in the question. Otherwise ROA = net income / average total "
        "assets for the period; preserve whether the requested output is a ratio or percent.",
    ),
    MetricGuidance(
        "margin",
        re.compile(r"\b(?:gross|net profit|operating income)\s+(?:%\s+)?margin\b", re.IGNORECASE),
        "A margin is the named profit measure / revenue for the same period. For a multi-year "
        "average, calculate each annual margin first and then take their arithmetic mean.",
    ),
    MetricGuidance(
        "growth_rate",
        re.compile(r"\b(?:growth rate|grew|growth)\b", re.IGNORECASE),
        "Period growth = (new value - old value) / old value. Multiply by 100 only when the "
        "answer is requested as a percent.",
    ),
    MetricGuidance(
        "period_change",
        re.compile(r"\b(?:drop|decline|increase|decrease|change)\b", re.IGNORECASE),
        "For a numeric period comparison, report direction, beginning and ending values, "
        "and relative percentage change = (new - old) / old * 100. Also mention the "
        "absolute change when useful. Use the relative percentage as calculation_expression "
        "unless the question explicitly requests another unit.",
    ),
    MetricGuidance(
        "inventory_turnover",
        re.compile(r"\binventory turnover\b", re.IGNORECASE),
        "Follow any formula in the question. Otherwise inventory turnover = cost of goods "
        "sold / average inventory over the period.",
    ),
    MetricGuidance(
        "operating_cash_flow_ratio",
        re.compile(r"\boperating cash flow ratio\b", re.IGNORECASE),
        "Operating cash flow ratio = cash from operations / total current liabilities.",
    ),
)


def _safe_error_text(value: str, limit: int = 1000) -> str:
    compact = " ".join(value.split())
    redacted = _SECRET_PATTERN.sub("[REDACTED]", compact)
    return redacted[:limit]


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
    prompt_version: str
    definition_ids: tuple[str, ...]
    metric_definition: str | None
    selected_items: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApiProbeResult:
    name: str
    ok: bool
    detail: str


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


def financial_guidance(question: str) -> tuple[MetricGuidance, ...]:
    return tuple(item for item in _METRIC_GUIDANCE if item.trigger.search(question))


def build_analysis_prompt(question: str, evidence: Sequence[dict[str, Any]]) -> str:
    guidance = financial_guidance(question)
    guidance_text = "\n".join(
        f"- [{item.guidance_id}] {item.instruction}" for item in guidance
    ) or "- No named metric rule matched. Follow any definition stated in the question."
    return (
        f"Prompt protocol: {PROMPT_VERSION}. Answer using only the supplied PDF evidence.\n"
        "First identify the requested metric and its financial definition. A definition "
        "explicitly supplied by the question overrides the guidance below. Never use a "
        "gold answer, hidden label or outside company fact.\n\n"
        "Relevant financial guidance:\n"
        f"{guidance_text}\n\n"
        "Return one JSON object with keys: answer, citations, needs_calculation, "
        "metric_definition, selected_items, calculation_expression. selected_items must "
        "be a list of objects with label, value, role and citation. Values and line-item "
        "labels must be copied from evidence; role explains add/subtract/numerator/"
        "denominator/context. Citations must be integer evidence IDs.\n"
        "If arithmetic is required, set needs_calculation=true, leave answer empty, and "
        "provide one Python-style arithmetic expression containing numbers and + - * / "
        "** parentheses only. The percent sign is not an operator; use * 100 for a percent. "
        "Do not calculate mentally. Keep source units; multiply by "
        "100 only for percent output. If no arithmetic is needed, set needs_calculation=false "
        "and answer directly. If the required line items are missing, say evidence is "
        "insufficient and use no citations.\n\n"
        f"Question: {question}\nEvidence:\n"
        f"{json.dumps(list(evidence), ensure_ascii=False)}"
    )


def build_calculated_prompt(
    question: str,
    evidence: Sequence[dict[str, Any]],
    expression: str,
    result: float,
    analysis: dict[str, Any] | None = None,
) -> str:
    return (
        "Produce the final concise answer using only the evidence and verified "
        "calculator output. Return JSON with keys answer and citations. Preserve the "
        "requested units and rounding. Do not replace the calculator result with mental "
        "arithmetic or invent units, thresholds or facts.\n\n"
        f"Question: {question}\n"
        f"Selected definition and inputs: {json.dumps(analysis or {}, ensure_ascii=False)}\n"
        f"Verified calculator: {expression} = {result}\nEvidence:\n"
        f"{json.dumps(list(evidence), ensure_ascii=False)}"
    )


_WORKING_CAPITAL_LINE_ITEMS = (
    ("cash and cash equivalents", re.compile(r"\bcash and cash equivalents\b", re.IGNORECASE)),
    ("trade accounts receivable", re.compile(r"\btrade accounts receivable\b", re.IGNORECASE)),
    ("accounts receivable", re.compile(r"(?<!trade )\baccounts receivable\b", re.IGNORECASE)),
    ("inventories", re.compile(r"\binventor(?:y|ies)\b", re.IGNORECASE)),
    ("other current assets", re.compile(r"\bother current assets\b", re.IGNORECASE)),
    ("income tax receivable", re.compile(r"\bincome tax receivable\b", re.IGNORECASE)),
    ("unbilled revenues", re.compile(r"\bunbilled revenues?\b", re.IGNORECASE)),
    ("materials and supplies", re.compile(r"\bmaterials and supplies\b", re.IGNORECASE)),
    ("loans and interest receivable", re.compile(r"\bloans and interest receivable\b", re.IGNORECASE)),
    ("funds receivable", re.compile(r"\bfunds receivable\b", re.IGNORECASE)),
    ("current portion of long-term debt", re.compile(r"\bcurrent portion of long-term debt\b", re.IGNORECASE)),
    ("accounts payable", re.compile(r"\baccounts payable\b", re.IGNORECASE)),
    ("accrued liabilities", re.compile(r"\b(?:other )?accrued liabilities\b", re.IGNORECASE)),
    ("accrued taxes", re.compile(r"\baccrued taxes\b", re.IGNORECASE)),
    ("funds payable", re.compile(r"\bfunds payable\b", re.IGNORECASE)),
    ("income taxes payable", re.compile(r"\bincome taxes payable\b", re.IGNORECASE)),
)


def analysis_plan_issues(
    question: str,
    analysis: dict[str, Any],
    evidence: Sequence[dict[str, Any]] = (),
) -> tuple[str, ...]:
    """Validate a model plan using only the question and its proposed inputs."""
    issues: list[str] = []
    guidance_ids = {item.guidance_id for item in financial_guidance(question)}
    raw_items = analysis.get("selected_items", [])
    items = [item for item in raw_items if isinstance(item, dict)] if isinstance(raw_items, list) else []
    labels = " | ".join(str(item.get("label", "")).casefold() for item in items)
    if "operating_working_capital" in guidance_ids:
        forbidden = (
            ("total current assets", False),
            ("total current liabilities", False),
            ("cash and cash equivalents", True),
            ("short-term investments", True),
            ("short term investments", True),
            ("short-term borrowings", True),
            ("short term borrowings", True),
            ("current portion of long-term debt", True),
        )
        used = []
        for term, allow_exclude in forbidden:
            for item in items:
                label = str(item.get("label", "")).casefold()
                role = str(item.get("role", "")).casefold()
                if term in label and not (allow_exclude and role == "exclude"):
                    used.append(term)
                    break
        if used:
            issues.append(
                "Operating working capital cannot use these proposed aggregate/non-operating "
                f"items: {', '.join(used)}. Select individual non-cash operating current "
                "assets and non-debt operating current liabilities from the evidence."
            )
        evidence_text = " ".join(str(item.get("text", "")) for item in evidence)
        missing = [
            name
            for name, pattern in _WORKING_CAPITAL_LINE_ITEMS
            if pattern.search(evidence_text) and not pattern.search(labels)
        ]
        if missing:
            issues.append(
                "The working-capital audit omitted line items visible in evidence: "
                f"{', '.join(missing)}. Add each to selected_items and mark it add, "
                "subtract or exclude before building the expression."
            )
    comparison = re.search(
        r"\b(?:drop|decline|increase|decrease|change|growth|grew)\b", question, re.IGNORECASE
    )
    if comparison and len(items) >= 2 and not bool(analysis.get("needs_calculation")):
        issues.append(
            "The question compares numeric periods. Quantify the absolute or percentage change "
            "with calculation_expression instead of returning only yes/no."
        )
    if "period_change" in guidance_ids and bool(analysis.get("needs_calculation")):
        expression = str(analysis.get("calculation_expression", ""))
        if "* 100" not in expression and "*100" not in expression:
            issues.append(
                "For this numeric period comparison, calculation_expression must compute the "
                "relative percentage change with * 100; the final answer may also state the "
                "absolute change."
            )
    return tuple(issues)


def build_plan_revision_prompt(
    question: str,
    evidence: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    issues: Sequence[str],
) -> str:
    return (
        "Revise the proposed financial analysis plan. The checks below are deterministic "
        "domain rules, not gold-answer feedback. Return the same JSON schema as the analysis "
        "request. Do not defend the old plan. Select only values present in evidence.\n\n"
        f"Question: {question}\n"
        f"Rejected plan: {json.dumps(analysis, ensure_ascii=False)}\n"
        f"Issues: {json.dumps(list(issues), ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(list(evidence), ensure_ascii=False)}"
    )


def _default_post(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
    *,
    trust_env_proxy: bool = True,
) -> dict[str, Any]:
    if trust_env_proxy:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    else:
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout)
        finally:
            session.close()
    if not response.ok:
        raw_text = _safe_error_text(response.text)
        try:
            error_payload = response.json()
        except requests.exceptions.JSONDecodeError:
            error_payload = {}
        if not isinstance(error_payload, dict):
            error_payload = {}
        error = error_payload.get("error", error_payload)
        if isinstance(error, dict):
            code = error.get("code") or error_payload.get("code")
            message = error.get("message") or error_payload.get("message")
            request_id = error_payload.get("request_id") or error_payload.get("requestId")
        else:
            code = None
            message = str(error) if error else None
            request_id = None
        request_id = request_id or response.headers.get("x-request-id") or response.headers.get(
            "x-dashscope-request-id"
        )
        details = [f"HTTP {response.status_code}"]
        if code:
            details.append(f"code={code}")
        if message:
            details.append(f"message={message}")
        if request_id:
            details.append(f"request_id={request_id}")
        content_type = response.headers.get("content-type")
        if content_type:
            details.append(f"content_type={content_type}")
        if raw_text and not message:
            details.append(f"body={raw_text}")
        raise RuntimeError("model API request failed: " + "; ".join(details))
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


def validate_api_credentials(base_url: str, api_key: str, model: str) -> None:
    if not base_url.strip() or not api_key.strip() or not model.strip():
        raise ValueError("base_url, api_key, and model are required")
    if base_url != base_url.strip():
        raise ValueError("base URL contains leading or trailing whitespace")
    if api_key != api_key.strip():
        raise ValueError("API Key contains leading or trailing whitespace")
    if model != model.strip():
        raise ValueError("model contains leading or trailing whitespace")
    if any(character.isspace() for character in api_key):
        raise ValueError("API Key contains whitespace")


def probe_openai_compatible_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 90,
    trust_env_proxy: bool = True,
) -> list[ApiProbeResult]:
    """Run progressively stricter calls without exposing the credential."""
    validate_api_credentials(base_url, api_key, model)
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    checks = (
        (
            "minimal_chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
            },
        ),
        (
            "non_thinking_chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "enable_thinking": False,
            },
        ),
        (
            "json_mode",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": 'Return JSON exactly like {"status":"ok"}.'}
                ],
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            },
        ),
    )
    post_json = partial(_default_post, trust_env_proxy=trust_env_proxy)
    results: list[ApiProbeResult] = []
    for name, payload in checks:
        try:
            response = post_json(endpoint, headers, payload, timeout)
        except Exception as error:
            results.append(ApiProbeResult(name=name, ok=False, detail=_safe_error_text(str(error))))
            break
        request_id = str(response.get("request_id") or response.get("id") or "not_returned")
        results.append(ApiProbeResult(name=name, ok=True, detail=f"request_id={request_id}"))
    return results


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
        request_options: dict[str, Any] | None = None,
        trust_env_proxy: bool = True,
        post_json: PostJson | None = None,
    ) -> None:
        validate_api_credentials(base_url, api_key, model)
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._timeout = timeout
        self._provider = provider
        self._request_options = dict(request_options or {})
        self._trust_env_proxy = trust_env_proxy
        self._post_json = post_json or partial(
            _default_post,
            trust_env_proxy=trust_env_proxy,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self._provider,
            "model": self._model,
            "endpoint": self._url,
            "trust_env_proxy": self._trust_env_proxy,
        }

    def _request(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a grounded financial QA system."},
                {"role": "user", "content": prompt},
            ],
        }
        payload.update(self._request_options)
        response = self._post_json(
            self._url,
            self._headers,
            payload,
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
        request_count = 1
        issues = analysis_plan_issues(question, first, evidence)
        if issues:
            revised, revision_usage = self._request(
                build_plan_revision_prompt(question, evidence, first, issues)
            )
            remaining_issues = analysis_plan_issues(question, revised, evidence)
            if remaining_issues:
                raise ValueError(
                    "generator returned an invalid financial plan after revision: "
                    + "; ".join(remaining_issues)
                )
            first = revised
            usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0))
                + int(revision_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0))
                + int(revision_usage.get("completion_tokens", 0)),
            }
            request_count += 1
        guidance = financial_guidance(question)
        metric_definition = str(first.get("metric_definition", "")).strip() or None
        raw_selected_items = first.get("selected_items", [])
        selected_items = tuple(
            dict(item) for item in raw_selected_items if isinstance(item, dict)
        ) if isinstance(raw_selected_items, list) else ()
        expression = None
        calculation_result = None
        final = first
        if bool(first.get("needs_calculation")):
            expression = str(first.get("calculation_expression", "")).strip()
            calculation_result = calculate(expression)
            final, second_usage = self._request(
                build_calculated_prompt(
                    question,
                    evidence,
                    expression,
                    calculation_result,
                    analysis={
                        "metric_definition": metric_definition,
                        "selected_items": list(selected_items),
                    },
                )
            )
            usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0))
                + int(second_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0))
                + int(second_usage.get("completion_tokens", 0)),
            }
            request_count += 1
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
            prompt_version=PROMPT_VERSION,
            definition_ids=tuple(item.guidance_id for item in guidance),
            metric_definition=metric_definition,
            selected_items=selected_items,
        )

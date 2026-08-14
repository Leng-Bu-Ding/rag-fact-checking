from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunk_io import read_chunks_jsonl
from src.data.financebench import read_questions_jsonl
from src.evaluation.answers import normalize_answer, token_f1
from src.evaluation.financebench import gold_pages
from src.generation.finance import (
    OpenAICompatibleFinanceGenerator,
    PROMPT_VERSION,
    build_analysis_prompt,
    evidence_payload,
    probe_openai_compatible_api,
)
from src.retrieval.types import RetrievalResult
from src.utils.config import load_config

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_PERCENT_RE = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*%")


def load_local_environment(path: Path) -> None:
    """Load a git-ignored dotenv file without overriding the active terminal."""
    if not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid local environment line {line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment variable name on line {line_number}")
        os.environ.setdefault(name, value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run FinanceBench evidence-grounded generation."
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "financebench.yaml")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--probe-api",
        action="store_true",
        help="Validate credentials and API features without reading or writing predictions.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--evaluation-set",
        choices=("dev", "holdout", "all"),
        default="dev",
        help="Use dev while changing prompts; reserve holdout for final validation.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def numbers(text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in _NUMBER_RE.findall(text)]


def numeric_match(prediction: str, reference: str) -> float:
    expected_percentages = [
        float(value.replace(",", "")) for value in _PERCENT_RE.findall(reference)
    ]
    if expected_percentages:
        predicted_percentages = [
            float(value.replace(",", "")) for value in _PERCENT_RE.findall(prediction)
        ]
        return float(
            all(
                any(abs(abs(actual) - abs(target)) <= 0.5 for actual in predicted_percentages)
                for target in expected_percentages
            )
        )
    predicted = numbers(prediction)
    expected = numbers(reference)
    if not expected:
        return 0.0
    return float(
        all(
            any(abs(actual - target) <= max(1e-4, abs(target) * 1e-3) for actual in predicted)
            for target in expected
        )
    )


def select_question_records(
    records: list[dict], evaluation_set: str, configured_sets: dict[str, list[str]]
) -> list[dict]:
    if evaluation_set == "all":
        return records
    requested = list(configured_sets.get(evaluation_set, []))
    if not requested:
        raise ValueError(f"evaluation set {evaluation_set!r} is empty")
    by_id = {str(item["financebench_id"]): item for item in records}
    missing = [question_id for question_id in requested if question_id not in by_id]
    if missing:
        raise ValueError(f"evaluation set contains unavailable questions: {missing}")
    return [by_id[question_id] for question_id in requested]


def generation_output_path(generation: dict, kind: str, evaluation_set: str) -> Path:
    template = str(generation[f"{kind}_path_template"])
    return project_path(template.format(evaluation_set=evaluation_set))


def rescore_record(record: dict) -> None:
    prediction = str(record.get("prediction", {}).get("answer", ""))
    reference = str(record.get("reference", ""))
    metrics = record.setdefault("metrics", {})
    metrics["exact_match"] = float(
        normalize_answer(prediction) == normalize_answer(reference)
    )
    metrics["token_f1"] = token_f1(prediction, reference)
    metrics["numeric_match"] = numeric_match(prediction, reference)


def main() -> None:
    args = parse_args()
    load_local_environment(PROJECT_ROOT / ".env.local")
    settings = load_config(args.config)["financebench"]
    generation = settings["generation"]
    report = json.loads(
        project_path(settings["retrieval_output_path"]).read_text(encoding="utf-8")
    )
    scope = str(generation["retrieval_scope"])
    system = str(generation["retrieval_system"])
    retrieval = report["scopes"][scope]["systems"][system]
    question_records = select_question_records(
        list(retrieval["questions"]),
        args.evaluation_set,
        dict(generation.get("evaluation_sets", {})),
    )
    if args.limit is not None:
        question_records = question_records[: args.limit]
    questions = {
        item.financebench_id: item
        for item in read_questions_jsonl(project_path(settings["questions_path"]))
    }
    chunks = {
        item.chunk_id: item
        for item in read_chunks_jsonl(project_path(settings["chunks_path"]))
    }
    top_k = int(generation["top_k"])

    prepared = []
    ranking_map: dict[str, list[RetrievalResult]] = {}
    for record in question_records:
        results = [
            RetrievalResult(
                score=float(item["score"]),
                rank=int(item["rank"]),
                chunk=chunks[item["chunk_id"]],
            )
            for item in record["results"][:top_k]
        ]
        ranking_map[record["financebench_id"]] = results
        evidence = evidence_payload(results, top_k)
        prompt = build_analysis_prompt(record["question"], evidence)
        prepared.append(
            {
                "financebench_id": record["financebench_id"],
                "evidence_count": len(evidence),
                "prompt_characters": len(prompt),
                "all_pages_are_real_pdf_pages": all(item["page"] is not None for item in evidence),
            }
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "evaluation_set": args.evaluation_set,
                    "prompt_version": PROMPT_VERSION,
                    "scope": scope,
                    "system": system,
                    "question_count": len(prepared),
                    "top_k": top_k,
                    "all_pages_are_real_pdf_pages": all(
                        item["all_pages_are_real_pdf_pages"] for item in prepared
                    ),
                    "average_prompt_characters": round(
                        sum(item["prompt_characters"] for item in prepared) / len(prepared), 2
                    ),
                    "note": "No API request was sent and no answer metric was fabricated.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    base_url = os.environ.get(str(generation["base_url_env"]), "")
    api_key = os.environ.get(str(generation["api_key_env"]), "")
    model = os.environ.get(str(generation["model_env"]), "")
    if not base_url or not api_key or not model:
        raise ValueError(
            "generation requires environment variables "
            f"{generation['base_url_env']}, {generation['api_key_env']}, and {generation['model_env']}"
        )
    if args.probe_api:
        probe = probe_openai_compatible_api(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=int(generation["timeout_seconds"]),
            trust_env_proxy=bool(generation.get("trust_env_proxy", True)),
        )
        print(
            json.dumps(
                {
                    "endpoint": f"{base_url.rstrip('/')}/chat/completions",
                    "model": model,
                    "key_configured": True,
                    "checks": [
                        {"name": item.name, "ok": item.ok, "detail": item.detail}
                        for item in probe
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if not probe or not all(item.ok for item in probe):
            raise SystemExit(2)
        return
    generator = OpenAICompatibleFinanceGenerator(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=int(generation["timeout_seconds"]),
        provider=str(generation["provider"]),
        request_options=dict(generation.get("request_options", {})),
        trust_env_proxy=bool(generation.get("trust_env_proxy", True)),
    )
    predictions_path = generation_output_path(
        generation, "predictions", args.evaluation_set
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    records: list[dict] = []
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            rescore_record(record)
            question_id = str(record["financebench_id"])
            question = questions[question_id]
            record.setdefault("question_type", question.question_type)
            pages = gold_pages(question)
            retrieved_pages = {
                (result.chunk.title, result.chunk.page_number)
                for result in ranking_map[question_id]
            }
            record.setdefault(
                "all_gold_pages_retrieved",
                bool(pages) and pages.issubset(retrieved_pages),
            )
            completed.add(record["financebench_id"])
            records.append(record)
    with predictions_path.open("a", encoding="utf-8", newline="\n") as output:
        for item in question_records:
            question_id = item["financebench_id"]
            if question_id in completed:
                continue
            question = questions[question_id]
            results = ranking_map[question_id]
            pages = gold_pages(question)
            retrieved_pages = {
                (result.chunk.title, result.chunk.page_number) for result in results
            }
            evidence_retrieved = bool(pages) and pages.issubset(retrieved_pages)
            started = time.perf_counter()
            try:
                generated = generator.generate(
                    question.question, results, max_evidence=top_k
                )
            except Exception as error:
                elapsed = time.perf_counter() - started
                record = {
                    "financebench_id": question_id,
                    "evaluation_set": args.evaluation_set,
                    "prompt_version": PROMPT_VERSION,
                    "question": question.question,
                    "reference": question.answer,
                    "question_type": question.question_type,
                    "all_gold_pages_retrieved": evidence_retrieved,
                    "status": "generation_error",
                    "error": f"{type(error).__name__}: {error}",
                    "prediction": {},
                    "metrics": {
                        "exact_match": 0.0,
                        "token_f1": 0.0,
                        "numeric_match": 0.0,
                        "citation_precision": 0.0,
                        "citation_recall": 0.0,
                    },
                    "latency_seconds": round(elapsed, 6),
                    "evidence": evidence_payload(results, top_k),
                }
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
                records.append(record)
                continue
            elapsed = time.perf_counter() - started
            cited = [results[index - 1] for index in generated.citations]
            supporting = sum(
                (result.chunk.title, result.chunk.page_number) in pages for result in cited
            )
            covered = {
                (result.chunk.title, result.chunk.page_number)
                for result in cited
                if (result.chunk.title, result.chunk.page_number) in pages
            }
            record = {
                "financebench_id": question_id,
                "evaluation_set": args.evaluation_set,
                "prompt_version": PROMPT_VERSION,
                "question": question.question,
                "reference": question.answer,
                "status": "ok",
                "question_type": question.question_type,
                "all_gold_pages_retrieved": evidence_retrieved,
                "prediction": generated.to_dict(),
                "metrics": {
                    "exact_match": float(
                        normalize_answer(generated.answer) == normalize_answer(question.answer)
                    ),
                    "token_f1": token_f1(generated.answer, question.answer),
                    "numeric_match": numeric_match(generated.answer, question.answer),
                    "citation_precision": supporting / len(cited) if cited else 0.0,
                    "citation_recall": len(covered) / len(pages),
                },
                "latency_seconds": round(elapsed, 6),
                "evidence": evidence_payload(results, top_k),
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            records.append(record)
    metric_names = tuple(records[0]["metrics"]) if records else ()
    evidence_available = [item for item in records if item.get("all_gold_pages_retrieved")]
    by_question_type = {}
    for question_type in sorted({str(item.get("question_type", "unknown")) for item in records}):
        typed = [item for item in records if item.get("question_type", "unknown") == question_type]
        by_question_type[question_type] = {
            "question_count": len(typed),
            "numeric_match": round(
                sum(item["metrics"]["numeric_match"] for item in typed) / len(typed), 8
            ),
            "citation_recall": round(
                sum(item["metrics"]["citation_recall"] for item in typed) / len(typed), 8
            ),
        }
    summary = {
        "schema_version": 1,
        "question_count": len(records),
        "successful_questions": sum(item.get("status", "ok") == "ok" for item in records),
        "failed_questions": sum(item.get("status") == "generation_error" for item in records),
        "all_gold_pages_retrieved_questions": len(evidence_available),
        "evaluation_set": args.evaluation_set,
        "prompt_version": PROMPT_VERSION,
        "scope": scope,
        "system": system,
        "generator": generator.metadata,
        "metrics": {
            name: round(sum(item["metrics"][name] for item in records) / len(records), 8)
            for name in metric_names
        },
        "metrics_when_all_gold_pages_retrieved": {
            name: round(
                sum(item["metrics"][name] for item in evidence_available)
                / len(evidence_available),
                8,
            )
            for name in metric_names
        } if evidence_available else {},
        "by_question_type": by_question_type,
        "total_latency_seconds": round(sum(item["latency_seconds"] for item in records), 6),
        "prompt_tokens": sum(
            item["prediction"].get("prompt_tokens") or 0 for item in records
        ),
        "completion_tokens": sum(
            item["prediction"].get("completion_tokens") or 0 for item in records
        ),
    }
    summary_path = generation_output_path(generation, "summary", args.evaluation_set)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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
    build_analysis_prompt,
    evidence_payload,
)
from src.retrieval.types import RetrievalResult
from src.utils.config import load_config

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run FinanceBench evidence-grounded generation."
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "financebench.yaml")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def numbers(text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in _NUMBER_RE.findall(text)]


def numeric_match(prediction: str, reference: str) -> float:
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


def main() -> None:
    args = parse_args()
    settings = load_config(args.config)["financebench"]
    generation = settings["generation"]
    report = json.loads(
        project_path(settings["retrieval_output_path"]).read_text(encoding="utf-8")
    )
    scope = str(generation["retrieval_scope"])
    system = str(generation["retrieval_system"])
    retrieval = report["scopes"][scope]["systems"][system]
    question_records = retrieval["questions"]
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
    generator = OpenAICompatibleFinanceGenerator(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=int(generation["timeout_seconds"]),
        provider=str(generation["provider"]),
    )
    predictions_path = project_path(generation["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    records: list[dict] = []
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            completed.add(record["financebench_id"])
            records.append(record)
    with predictions_path.open("a", encoding="utf-8", newline="\n") as output:
        for item in question_records:
            question_id = item["financebench_id"]
            if question_id in completed:
                continue
            question = questions[question_id]
            results = ranking_map[question_id]
            started = time.perf_counter()
            generated = generator.generate(question.question, results, max_evidence=top_k)
            elapsed = time.perf_counter() - started
            pages = gold_pages(question)
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
                "question": question.question,
                "reference": question.answer,
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
    summary = {
        "schema_version": 1,
        "question_count": len(records),
        "scope": scope,
        "system": system,
        "generator": generator.metadata,
        "metrics": {
            name: round(sum(item["metrics"][name] for item in records) / len(records), 8)
            for name in metric_names
        },
        "total_latency_seconds": round(sum(item["latency_seconds"] for item in records), 6),
        "prompt_tokens": sum(
            item["prediction"].get("prompt_tokens") or 0 for item in records
        ),
        "completion_tokens": sum(
            item["prediction"].get("completion_tokens") or 0 for item in records
        ),
    }
    summary_path = project_path(generation["summary_path"])
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

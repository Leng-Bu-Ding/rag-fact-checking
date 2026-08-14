from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.financebench import summarize_finance_failures


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def finance_summary(report: dict[str, Any]) -> dict[str, Any]:
    scopes: dict[str, Any] = {}
    for scope_name, scope in report["scopes"].items():
        systems: dict[str, Any] = {}
        for system_name, evaluation in scope["systems"].items():
            systems[system_name] = {
                "metrics": evaluation["metrics"],
                "failure_analysis_at_10": summarize_finance_failures(
                    evaluation, k=10
                ),
            }
        scopes[scope_name] = {
            "systems": systems,
            "timings": scope["timings"],
            "reranker": scope["reranker"],
        }
    return {
        "schema_version": 1,
        "experiment": report["experiment"],
        "source": report["source"],
        "coverage": report["coverage"],
        "leakage_checks": report["leakage_checks"],
        "shared_timings": report["shared_timings"],
        "scopes": scopes,
        "limitations": [
            "Only questions whose source PDF passed download and exact page-alignment validation are evaluated.",
            "Cross-Encoder results use a generic MS MARCO model without financial-domain fine-tuning.",
        ],
    }


def finance_generation_summary(
    generation: dict[str, Any], judge: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "financebench_api_generation_v5",
        "evaluation": {
            "question_count": generation["question_count"],
            "retrieval_scope": generation["scope"],
            "retrieval_system": generation["system"],
            "prompt_version": generation["prompt_version"],
            "generator_model": generation["generator"]["model"],
            "successful_questions": generation["successful_questions"],
            "failed_questions": generation["failed_questions"],
            "all_gold_pages_retrieved_questions": generation[
                "all_gold_pages_retrieved_questions"
            ],
        },
        "deterministic_metrics": generation["metrics"],
        "deterministic_metrics_when_all_gold_pages_retrieved": generation[
            "metrics_when_all_gold_pages_retrieved"
        ],
        "generation_by_question_type": generation["by_question_type"],
        "answer_judge": {
            "judge_version": judge["judge_version"],
            "judge_model": judge["judge_model"],
            "verdicts": judge["verdicts"],
            "overall_correct_rate": judge["overall_correct_rate"],
            "generated_answer_correct_rate": judge["generated_answer_correct_rate"],
            "correct_rate_when_all_gold_pages_retrieved": judge[
                "correct_rate_when_all_gold_pages_retrieved"
            ],
            "by_question_type": judge["by_question_type"],
            "error_types": judge["error_types"],
        },
        "usage": {
            "generation_prompt_tokens": generation["prompt_tokens"],
            "generation_completion_tokens": generation["completion_tokens"],
            "judge_prompt_tokens": judge["prompt_tokens"],
            "judge_completion_tokens": judge["completion_tokens"],
            "generation_latency_seconds": generation["total_latency_seconds"],
            "judge_latency_seconds": judge["total_latency_seconds"],
        },
        "limitations": [
            "The 114 questions are the subset whose PDFs passed download and exact page-alignment validation; this is not 150/150 coverage.",
            "Exact match underestimates semantically equivalent free-form and rounded numerical answers.",
            "The answer judge uses the same model family as the generator and is not an independent human audit.",
            "Gold evidence and justifications are evaluation-only and are never included in generation prompts or the retrieval index.",
            "Monetary cost is not inferred from token counts because the run used provider free quota and the billing statement was not imported.",
        ],
    }


def hotpot_generation_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "hotpotqa_grounded_generation_after_citation_fix",
        "dataset": report["dataset"],
        "metrics": report["metrics"],
        "timing": report["timing"],
        "citation_policy": "model_generated_then_validated; no forced citation injection",
        "limitations": [
            "The local FLAN-T5-small generator emits valid citations for only a minority of answers.",
            "Citation validity is not equivalent to claim-level faithfulness.",
        ],
    }


def hotpot_retrieval_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": report["experiment"],
        "source": report["source"],
        "split": {
            "seed": report["split"]["seed"],
            "sha256": report["split"]["sha256"],
            "dev_count": len(report["split"]["dev"]),
            "test_count": len(report["split"]["test"]),
        },
        "systems": {
            name: {"metrics": value["metrics"]}
            for name, value in report["systems"].items()
        },
        "timings": report["timings"],
        "reranker": report["reranker"],
        "tuning": {"selected": report["tuning"]["selected"]},
    }


def main() -> None:
    public_dir = PROJECT_ROOT / "results" / "public"
    finance = read_json(PROJECT_ROOT / "results" / "financebench_retrieval.json")
    hotpot_retrieval = read_json(
        PROJECT_ROOT / "results" / "hotpotqa_retrieval_core_sample100.json"
    )
    hotpot = read_json(
        PROJECT_ROOT / "results" / "hotpotqa_generation_sample100_test80.json"
    )
    finance_generation = read_json(
        PROJECT_ROOT / "results" / "financebench_generation_all_v5_summary.json"
    )
    finance_judge = read_json(
        PROJECT_ROOT / "results" / "financebench_answer_judge_all_v1_summary.json"
    )
    write_json(public_dir / "financebench_retrieval_summary.json", finance_summary(finance))
    write_json(
        public_dir / "hotpotqa_retrieval_summary.json",
        hotpot_retrieval_summary(hotpot_retrieval),
    )
    write_json(public_dir / "hotpotqa_generation_summary.json", hotpot_generation_summary(hotpot))
    write_json(
        public_dir / "financebench_generation_summary.json",
        finance_generation_summary(finance_generation, finance_judge),
    )
    print(f"Published summaries to {public_dir}")


if __name__ == "__main__":
    main()

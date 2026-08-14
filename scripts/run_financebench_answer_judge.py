from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_financebench_generation_evaluation import load_local_environment
from src.evaluation.finance_answers import (
    JUDGE_VERSION,
    OpenAICompatibleFinanceAnswerJudge,
)
from src.utils.config import load_config

_ABSTENTION_MARKERS = ("evidence is insufficient", "cannot answer", "not enough evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge frozen FinanceBench predictions.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "financebench.yaml")
    )
    parser.add_argument("--evaluation-set", choices=("dev", "holdout", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def output_path(generation: dict, kind: str, evaluation_set: str) -> Path:
    template = str(generation[f"judge_{kind}_path_template"])
    return project_path(template.format(evaluation_set=evaluation_set))


def main() -> None:
    args = parse_args()
    load_local_environment(PROJECT_ROOT / ".env.local")
    generation = load_config(args.config)["financebench"]["generation"]
    predictions_path = project_path(
        str(generation["predictions_path_template"]).format(
            evaluation_set=args.evaluation_set
        )
    )
    predictions = [
        json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    if args.limit is not None:
        predictions = predictions[: args.limit]

    base_url = os.environ.get(str(generation["base_url_env"]), "")
    api_key = os.environ.get(str(generation["api_key_env"]), "")
    model = os.environ.get(str(generation["model_env"]), "")
    judge = OpenAICompatibleFinanceAnswerJudge(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=int(generation["timeout_seconds"]),
        request_options=dict(generation.get("request_options", {})),
        trust_env_proxy=bool(generation.get("trust_env_proxy", True)),
    )
    judgments_path = output_path(generation, "predictions", args.evaluation_set)
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    judgments: list[dict] = []
    completed: set[str] = set()
    if judgments_path.exists():
        judgments = [
            json.loads(line)
            for line in judgments_path.read_text(encoding="utf-8").splitlines()
        ]
        completed = {str(item["financebench_id"]) for item in judgments}

    with judgments_path.open("a", encoding="utf-8", newline="\n") as output:
        for record in predictions:
            question_id = str(record["financebench_id"])
            if question_id in completed:
                continue
            prediction = str(record.get("prediction", {}).get("answer", ""))
            common = {
                "financebench_id": question_id,
                "question_type": record.get("question_type", "unknown"),
                "all_gold_pages_retrieved": bool(record.get("all_gold_pages_retrieved")),
                "judge_version": JUDGE_VERSION,
            }
            if record.get("status") == "generation_error":
                judgment = {**common, "verdict": "generation_error"}
            elif any(marker in prediction.casefold() for marker in _ABSTENTION_MARKERS):
                judgment = {**common, "verdict": "abstain"}
            else:
                started = time.perf_counter()
                try:
                    judged = judge.judge(
                        str(record["question"]), str(record["reference"]), prediction
                    )
                    judgment = {
                        **common,
                        "verdict": judged.verdict,
                        "error_type": judged.error_type,
                        "reason": judged.reason,
                        "model": judged.model,
                        "prompt_tokens": judged.prompt_tokens,
                        "completion_tokens": judged.completion_tokens,
                        "latency_seconds": round(time.perf_counter() - started, 6),
                    }
                except Exception as error:
                    judgment = {
                        **common,
                        "verdict": "judge_error",
                        "error": f"{type(error).__name__}: {error}",
                        "latency_seconds": round(time.perf_counter() - started, 6),
                    }
            output.write(json.dumps(judgment, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            judgments.append(judgment)

    verdicts = Counter(str(item["verdict"]) for item in judgments)
    evidence_available = [item for item in judgments if item["all_gold_pages_retrieved"]]
    by_question_type = {}
    for question_type in sorted({str(item["question_type"]) for item in judgments}):
        typed = [item for item in judgments if item["question_type"] == question_type]
        typed_verdicts = Counter(str(item["verdict"]) for item in typed)
        by_question_type[question_type] = {
            "question_count": len(typed),
            "correct_rate": round(typed_verdicts["correct"] / len(typed), 8),
            "verdicts": dict(sorted(typed_verdicts.items())),
        }
    summary = {
        "schema_version": 1,
        "judge_version": JUDGE_VERSION,
        "judge_model": judge.model,
        "evaluation_set": args.evaluation_set,
        "question_count": len(judgments),
        "verdicts": dict(sorted(verdicts.items())),
        "overall_correct_rate": round(verdicts["correct"] / len(judgments), 8),
        "generated_answer_correct_rate": round(
            verdicts["correct"]
            / max(1, verdicts["correct"] + verdicts["incorrect"]),
            8,
        ),
        "all_gold_pages_retrieved_question_count": len(evidence_available),
        "correct_rate_when_all_gold_pages_retrieved": round(
            sum(item["verdict"] == "correct" for item in evidence_available)
            / max(1, len(evidence_available)),
            8,
        ),
        "error_types": dict(
            sorted(Counter(str(item.get("error_type", "")) for item in judgments if item.get("error_type") and item["error_type"] != "none").items())
        ),
        "by_question_type": by_question_type,
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in judgments),
        "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in judgments),
        "total_latency_seconds": round(
            sum(float(item.get("latency_seconds") or 0) for item in judgments), 6
        ),
        "limitations": [
            "The judge sees only question, reference answer, and candidate answer.",
            "The judge model is the same model family as the generator, so results are not an independent human audit.",
        ],
    }
    summary_path = output_path(generation, "summary", args.evaluation_set)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

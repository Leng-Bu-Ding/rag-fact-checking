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
            "Answer-generation accuracy is not reported until an API-backed run is completed.",
            "Cross-Encoder results use a generic MS MARCO model without financial-domain fine-tuning.",
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
    write_json(public_dir / "financebench_retrieval_summary.json", finance_summary(finance))
    write_json(
        public_dir / "hotpotqa_retrieval_summary.json",
        hotpot_retrieval_summary(hotpot_retrieval),
    )
    write_json(public_dir / "hotpotqa_generation_summary.json", hotpot_generation_summary(hotpot))
    print(f"Published summaries to {public_dir}")


if __name__ == "__main__":
    main()

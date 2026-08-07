from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunk_io import read_chunks_jsonl
from src.evaluation.answers import evaluate_answer
from src.evaluation.retrieval import aggregate_metrics, gold_facts_for_sample
from src.generation.grounded import LocalGroundedGenerator
from src.pipelines.bm25 import group_chunks_by_sample
from src.pipelines.dense import sha256_file
from src.pipelines.retrieval_experiment import write_report
from src.retrieval.types import RetrievalResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded answers over fixed HotpotQA rankings."
    )
    parser.add_argument(
        "--retrieval-report",
        default=str(
            PROJECT_ROOT / "results" / "hotpotqa_retrieval_core_sample100.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "results" / "hotpotqa_generation_sample100_test80.json"
        ),
    )
    parser.add_argument("--system", default="reranker")
    parser.add_argument("--max-evidence", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolved(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    if args.max_evidence <= 0:
        raise ValueError("max-evidence must be greater than zero")
    retrieval_path = resolved(args.retrieval_report)
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    if args.system not in retrieval["systems"]:
        raise ValueError(f"retrieval report has no system named {args.system}")
    chunks_path = resolved(retrieval["source"]["chunks_path"])
    chunks = read_chunks_jsonl(chunks_path)
    grouped = group_chunks_by_sample(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    query_records = retrieval["systems"][args.system]["queries"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be greater than zero")
        query_records = query_records[: args.limit]

    dense_config = retrieval["config"]["dense"]
    generator = LocalGroundedGenerator(
        cache_dir=dense_config.get("cache_folder"),
        local_files_only=bool(dense_config.get("local_files_only", False)),
    )
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, float]] = []
    started = time.perf_counter()
    for query in query_records:
        sample_id = query["sample_id"]
        sample_chunks = grouped[sample_id]
        reference = sample_chunks[0].answer
        evidence = [
            RetrievalResult(
                score=float(item["score"]),
                rank=int(item["rank"]),
                chunk=chunks_by_id[item["chunk_id"]],
            )
            for item in query["results"][: args.max_evidence]
        ]
        query_started = time.perf_counter()
        prediction = generator.generate(
            query["question"], evidence, max_evidence=args.max_evidence
        )
        elapsed = time.perf_counter() - query_started
        query_metrics = evaluate_answer(
            prediction,
            reference,
            evidence,
            gold_facts_for_sample(sample_chunks, sample_id),
        )
        metrics.append(query_metrics)
        records.append(
            {
                "sample_id": sample_id,
                "question": query["question"],
                "reference_answer": reference,
                "prediction": prediction,
                "metrics": {
                    key: round(value, 8)
                    for key, value in sorted(query_metrics.items())
                },
                "generation_ms": round(elapsed * 1000, 3),
                "evidence_chunk_ids": [item.chunk.chunk_id for item in evidence],
            }
        )
    total_seconds = time.perf_counter() - started
    aggregate = aggregate_metrics(metrics)
    report = {
        "schema_version": 1,
        "experiment": "hotpotqa_grounded_generation",
        "source": {
            "retrieval_report": str(retrieval_path),
            "retrieval_report_sha256": sha256_file(retrieval_path),
            "retrieval_system": args.system,
            "split_sha256": retrieval["split"]["sha256"],
        },
        "config": {
            "max_evidence": args.max_evidence,
            "generator": generator.metadata,
        },
        "dataset": {"sample_count": len(records)},
        "metrics": {
            key: round(value, 8) for key, value in aggregate.items()
        },
        "timing": {
            "total_seconds": round(total_seconds, 6),
            "milliseconds_per_query": round(
                total_seconds * 1000 / len(records), 6
            ),
        },
        "queries": records,
    }
    output_path = resolved(args.output)
    write_report(report, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "dataset": report["dataset"],
                "metrics": report["metrics"],
                "timing": report["timing"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

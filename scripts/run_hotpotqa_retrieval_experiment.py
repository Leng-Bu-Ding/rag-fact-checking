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
from src.pipelines.bm25 import group_chunks_by_sample, question_for_sample
from src.pipelines.dense import sha256_file
from src.pipelines.retrieval_experiment import (
    compare_systems,
    deterministic_split,
    evaluate_rankings,
    fuse_rankings,
    split_sha256,
    timed_rerank,
    tune_hybrid,
    write_report,
)
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex, SentenceTransformerEncoder
from src.retrieval.reranker import (
    CrossEncoderReranker,
    TransformersCrossEncoderScorer,
)
from src.retrieval.types import RetrievalResult
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible HotpotQA retrieval comparison."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "hotpotqa_retrieval.yaml"),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Evaluate BM25, dense, and hybrid without loading a cross-encoder.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_base_rankings(
    chunks: list[Any],
    index: DenseIndex,
    encoder: SentenceTransformerEncoder,
    sample_ids: list[str],
    *,
    ranking_depth: int,
    dense_batch_size: int,
    bm25_config: dict[str, Any],
) -> tuple[
    dict[str, list[RetrievalResult]],
    dict[str, list[RetrievalResult]],
    dict[str, str],
    dict[str, dict[str, float]],
]:
    grouped = group_chunks_by_sample(chunks)
    questions = {
        sample_id: question_for_sample(grouped[sample_id], sample_id)
        for sample_id in sample_ids
    }

    started = time.perf_counter()
    bm25_index = BM25Index(
        chunks,
        include_title=bool(bm25_config["include_title"]),
        k1=float(bm25_config["k1"]),
        b=float(bm25_config["b"]),
        epsilon=float(bm25_config["epsilon"]),
    )
    bm25_rankings = {
        sample_id: bm25_index.search(question, top_k=ranking_depth)
        for sample_id, question in questions.items()
    }
    bm25_seconds = time.perf_counter() - started

    started = time.perf_counter()
    vectors = encoder.encode_queries(
        [questions[sample_id] for sample_id in sample_ids],
        batch_size=dense_batch_size,
    )
    dense_rankings = {
        sample_id: index.search_vector(vector, top_k=ranking_depth)
        for sample_id, vector in zip(sample_ids, vectors, strict=True)
    }
    dense_seconds = time.perf_counter() - started

    count = len(sample_ids)
    timings = {
        "bm25": {
            "total_seconds": round(bm25_seconds, 6),
            "milliseconds_per_query": round(bm25_seconds * 1000 / count, 6),
        },
        "dense": {
            "total_seconds": round(dense_seconds, 6),
            "milliseconds_per_query": round(dense_seconds * 1000 / count, 6),
        },
    }
    return bm25_rankings, dense_rankings, questions, timings


def main() -> None:
    args = parse_args()
    settings = load_config(args.config)["hotpotqa_retrieval"]
    chunks_path = project_path(str(settings["chunks_path"]))
    index_dir = project_path(str(settings["dense_index_dir"]))
    output_path = project_path(args.output or str(settings["output_path"]))
    chunks = read_chunks_jsonl(chunks_path)
    index, index_manifest = DenseIndex.load(index_dir)
    chunks_hash = sha256_file(chunks_path)
    if index_manifest.get("source", {}).get("chunks_sha256") != chunks_hash:
        raise ValueError("dense index and configured chunks do not match")

    grouped = group_chunks_by_sample(chunks)
    sample_ids = sorted(grouped)
    split_config = settings["split"]
    split = deterministic_split(
        sample_ids,
        dev_count=int(split_config["dev_count"]),
        seed=int(split_config["seed"]),
    )
    evaluation_config = settings["evaluation"]
    top_ks = [int(value) for value in evaluation_config["top_ks"]]
    ranking_depth = int(evaluation_config["ranking_depth"])
    dense_config = settings["dense"]
    encoder = SentenceTransformerEncoder(
        str(dense_config["model_name"]),
        revision=dense_config.get("revision"),
        device=str(dense_config["device"]),
        cache_folder=dense_config.get("cache_folder"),
        local_files_only=bool(dense_config["local_files_only"]),
        seed=int(split_config["seed"]),
    )
    if encoder.metadata["dimension"] != index.dimension:
        raise ValueError("dense encoder and persisted index dimensions do not match")

    bm25_rankings, dense_rankings, questions, timings = build_base_rankings(
        chunks,
        index,
        encoder,
        sample_ids,
        ranking_depth=ranking_depth,
        dense_batch_size=int(dense_config["batch_size"]),
        bm25_config=settings["bm25"],
    )
    hybrid_config = settings["hybrid_search"]
    tuning = tune_hybrid(
        chunks,
        bm25_rankings,
        dense_rankings,
        split["dev"],
        candidate_ks=[int(value) for value in hybrid_config["candidate_ks"]],
        rrf_ks=[int(value) for value in hybrid_config["rrf_ks"]],
        top_ks=top_ks,
    )
    selected = tuning["selected"]
    started = time.perf_counter()
    hybrid_rankings = fuse_rankings(
        bm25_rankings,
        dense_rankings,
        split["test"],
        candidate_k=int(selected["candidate_k"]),
        rrf_k=int(selected["rrf_k"]),
        top_k=max(top_ks),
    )
    hybrid_seconds = time.perf_counter() - started
    timings["hybrid_fusion"] = {
        "total_seconds": round(hybrid_seconds, 6),
        "milliseconds_per_query": round(
            hybrid_seconds * 1000 / len(split["test"]), 6
        ),
    }

    evaluations = {
        "bm25": evaluate_rankings(
            chunks, bm25_rankings, split["test"], top_ks=top_ks
        ),
        "dense": evaluate_rankings(
            chunks, dense_rankings, split["test"], top_ks=top_ks
        ),
        "hybrid": evaluate_rankings(
            chunks, hybrid_rankings, split["test"], top_ks=top_ks
        ),
    }
    objective = f"recall_at_{max(top_ks)}"
    baseline = max(
        ("bm25", "dense"),
        key=lambda name: (evaluations[name]["metrics"][objective], name),
    )
    comparisons = {
        "hybrid_vs_best_single": compare_systems(
            evaluations,
            baseline=baseline,
            target="hybrid",
            metric=objective,
        )
    }

    reranker_record: dict[str, Any] | None = None
    reranker_config = settings["reranker"]
    if bool(reranker_config["enabled"]) and not args.skip_reranker:
        reranker_candidate_k = int(reranker_config["candidate_k"])
        reranker_candidates = fuse_rankings(
            bm25_rankings,
            dense_rankings,
            split["test"],
            candidate_k=max(
                reranker_candidate_k, int(selected["candidate_k"])
            ),
            rrf_k=int(selected["rrf_k"]),
            top_k=reranker_candidate_k,
        )
        scorer = TransformersCrossEncoderScorer(
            str(reranker_config["model_name"]),
            cache_dir=reranker_config.get("cache_dir"),
            device=str(reranker_config["device"]),
            local_files_only=bool(reranker_config["local_files_only"]),
            max_length=int(reranker_config["max_length"]),
        )
        reranker = CrossEncoderReranker(
            scorer,
            batch_size=int(reranker_config["batch_size"]),
        )
        reranked, reranker_timing = timed_rerank(
            reranker,
            {sample_id: questions[sample_id] for sample_id in split["test"]},
            reranker_candidates,
            top_k=max(top_ks),
        )
        evaluations["reranker"] = evaluate_rankings(
            chunks, reranked, split["test"], top_ks=top_ks
        )
        comparisons["reranker_vs_hybrid"] = compare_systems(
            evaluations,
            baseline="hybrid",
            target="reranker",
            metric=objective,
        )
        reranker_record = {
            "metadata": reranker.metadata,
            "candidate_k": reranker_candidate_k,
            "timing": reranker_timing,
        }

    report = {
        "schema_version": 1,
        "experiment": "hotpotqa_retrieval_core",
        "source": {
            "chunks_path": display_path(chunks_path),
            "chunks_sha256": chunks_hash,
            "dense_index_manifest": display_path(index_dir / "manifest.json"),
        },
        "split": {
            **split,
            "seed": int(split_config["seed"]),
            "sha256": split_sha256(split),
        },
        "config": settings,
        "tuning": tuning,
        "timings": timings,
        "reranker": reranker_record,
        "systems": evaluations,
        "comparisons": comparisons,
    }
    write_report(report, output_path)
    print(
        json.dumps(
            {
                "output": display_path(output_path),
                "split": {
                    "dev": len(split["dev"]),
                    "test": len(split["test"]),
                    "sha256": split_sha256(split),
                },
                "selected_hybrid": selected,
                "metrics": {
                    name: evaluation["metrics"]
                    for name, evaluation in evaluations.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

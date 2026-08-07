from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data.chunking import DocumentChunk
from src.evaluation.retrieval import (
    aggregate_metrics,
    covered_gold_facts,
    evaluate_query,
    gold_facts_for_sample,
)
from src.pipelines.bm25 import group_chunks_by_sample, question_for_sample
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.types import RetrievalResult

RankingMap = dict[str, list[RetrievalResult]]


def deterministic_split(
    sample_ids: Sequence[str],
    *,
    dev_count: int,
    seed: int,
) -> dict[str, list[str]]:
    unique_ids = sorted(set(sample_ids))
    if not 0 < dev_count < len(unique_ids):
        raise ValueError("dev_count must leave non-empty dev and test splits")
    ordered = sorted(
        unique_ids,
        key=lambda sample_id: (
            hashlib.sha256(f"{seed}|{sample_id}".encode()).hexdigest(),
            sample_id,
        ),
    )
    return {"dev": ordered[:dev_count], "test": ordered[dev_count:]}


def split_sha256(split: Mapping[str, Sequence[str]]) -> str:
    encoded = json.dumps(split, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def fuse_rankings(
    bm25: Mapping[str, Sequence[RetrievalResult]],
    dense: Mapping[str, Sequence[RetrievalResult]],
    sample_ids: Sequence[str],
    *,
    candidate_k: int,
    rrf_k: int,
    top_k: int,
) -> RankingMap:
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    return {
        sample_id: reciprocal_rank_fusion(
            {
                "bm25": bm25[sample_id][:candidate_k],
                "dense": dense[sample_id][:candidate_k],
            },
            top_k=top_k,
            rrf_k=rrf_k,
        )
        for sample_id in sample_ids
    }


def _result_record(
    result: RetrievalResult,
    gold_facts: frozenset[tuple[str, str, int]],
) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "score": round(result.score, 8),
        "chunk_id": result.chunk.chunk_id,
        "sample_id": result.chunk.sample_id,
        "doc_id": result.chunk.doc_id,
        "title": result.chunk.title,
        "sentence_ids": result.chunk.sentence_ids,
        "covered_gold_facts": [
            {"sample_id": sample, "title": title, "sentence_id": sentence}
            for sample, title, sentence in sorted(
                covered_gold_facts(result.chunk, gold_facts)
            )
        ],
    }


def evaluate_rankings(
    chunks: Sequence[DocumentChunk],
    rankings: Mapping[str, Sequence[RetrievalResult]],
    sample_ids: Sequence[str],
    *,
    top_ks: Sequence[int],
) -> dict[str, Any]:
    grouped = group_chunks_by_sample(chunks)
    ordered_ids = sorted(sample_ids)
    if not ordered_ids:
        raise ValueError("evaluation requires at least one sample")
    if not top_ks or any(k <= 0 for k in top_ks):
        raise ValueError("top_ks must contain positive integers")
    if any(sample_id not in grouped for sample_id in ordered_ids):
        raise ValueError("evaluation contains unknown sample IDs")
    if any(sample_id not in rankings for sample_id in ordered_ids):
        raise ValueError("rankings are missing evaluated sample IDs")

    ks = sorted(set(top_ks))
    query_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, float]] = []
    total_gold = 0
    for sample_id in ordered_ids:
        gold = gold_facts_for_sample(grouped[sample_id], sample_id)
        if not gold:
            raise ValueError(f"sample has no represented gold facts: {sample_id}")
        total_gold += len(gold)
        metrics = evaluate_query(rankings[sample_id], gold, ks=ks)
        metric_records.append(metrics)
        query_records.append(
            {
                "sample_id": sample_id,
                "question": question_for_sample(grouped[sample_id], sample_id),
                "gold_fact_count": len(gold),
                "metrics": {
                    key: round(value, 8)
                    for key, value in sorted(metrics.items())
                },
                "results": [
                    _result_record(result, gold)
                    for result in rankings[sample_id][: max(ks)]
                ],
            }
        )
    return {
        "dataset": {
            "sample_count": len(ordered_ids),
            "chunk_count": len(chunks),
            "gold_fact_count": total_gold,
        },
        "metrics": {
            key: round(value, 8)
            for key, value in aggregate_metrics(metric_records).items()
        },
        "queries": query_records,
    }


def tune_hybrid(
    chunks: Sequence[DocumentChunk],
    bm25: Mapping[str, Sequence[RetrievalResult]],
    dense: Mapping[str, Sequence[RetrievalResult]],
    dev_ids: Sequence[str],
    *,
    candidate_ks: Sequence[int],
    rrf_ks: Sequence[int],
    top_ks: Sequence[int],
) -> dict[str, Any]:
    if not candidate_ks or not rrf_ks:
        raise ValueError("hybrid search grid cannot be empty")
    max_k = max(top_ks)
    trials: list[dict[str, Any]] = []
    for candidate_k in sorted(set(candidate_ks)):
        for rrf_k in sorted(set(rrf_ks)):
            rankings = fuse_rankings(
                bm25,
                dense,
                dev_ids,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                top_k=max_k,
            )
            evaluation = evaluate_rankings(
                chunks, rankings, dev_ids, top_ks=top_ks
            )
            trials.append(
                {
                    "candidate_k": candidate_k,
                    "rrf_k": rrf_k,
                    "metrics": evaluation["metrics"],
                }
            )
    objective_k = max(top_ks)
    trials.sort(
        key=lambda trial: (
            -trial["metrics"][f"recall_at_{objective_k}"],
            -trial["metrics"][f"complete_at_{objective_k}"],
            -trial["metrics"][f"fact_ndcg_at_{objective_k}"],
            -trial["metrics"]["mrr"],
            trial["candidate_k"],
            trial["rrf_k"],
        )
    )
    return {"selected": trials[0], "trials": trials}


def compare_systems(
    evaluations: Mapping[str, dict[str, Any]],
    *,
    baseline: str,
    target: str,
    metric: str,
) -> dict[str, Any]:
    baseline_queries = {
        item["sample_id"]: item for item in evaluations[baseline]["queries"]
    }
    target_queries = {
        item["sample_id"]: item for item in evaluations[target]["queries"]
    }
    if set(baseline_queries) != set(target_queries):
        raise ValueError("systems must evaluate identical sample IDs")

    improved: list[str] = []
    regressed: list[str] = []
    tied: list[str] = []
    deltas: list[tuple[float, str]] = []
    for sample_id in sorted(baseline_queries):
        delta = (
            target_queries[sample_id]["metrics"][metric]
            - baseline_queries[sample_id]["metrics"][metric]
        )
        deltas.append((delta, sample_id))
        if delta > 1e-12:
            improved.append(sample_id)
        elif delta < -1e-12:
            regressed.append(sample_id)
        else:
            tied.append(sample_id)
    return {
        "baseline": baseline,
        "target": target,
        "metric": metric,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "tied_count": len(tied),
        "largest_improvements": [
            {"sample_id": sample_id, "delta": round(delta, 8)}
            for delta, sample_id in sorted(deltas, reverse=True)[:10]
            if delta > 0
        ],
        "largest_regressions": [
            {"sample_id": sample_id, "delta": round(delta, 8)}
            for delta, sample_id in sorted(deltas)[:10]
            if delta < 0
        ],
    }


def timed_rerank(
    reranker: CrossEncoderReranker,
    questions: Mapping[str, str],
    candidates: Mapping[str, Sequence[RetrievalResult]],
    *,
    top_k: int,
) -> tuple[RankingMap, dict[str, float]]:
    started = time.perf_counter()
    rankings = reranker.rerank_many(questions, candidates, top_k=top_k)
    elapsed = time.perf_counter() - started
    return rankings, {
        "total_seconds": round(elapsed, 6),
        "milliseconds_per_query": round(
            elapsed * 1000 / len(questions), 6
        ),
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

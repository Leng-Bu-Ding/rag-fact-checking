from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from src.data.chunking import DocumentChunk
from src.evaluation.retrieval import (
    aggregate_metrics,
    covered_gold_facts,
    evaluate_query,
    gold_facts_for_sample,
)
from src.retrieval.bm25 import BM25Index
from src.retrieval.types import RetrievalResult

CorpusScope = Literal["global", "sample"]


def group_chunks_by_sample(
    chunks: Sequence[DocumentChunk],
) -> dict[str, list[DocumentChunk]]:
    grouped: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.sample_id].append(chunk)
    return dict(grouped)


def question_for_sample(chunks: Sequence[DocumentChunk], sample_id: str) -> str:
    questions = {
        chunk.question for chunk in chunks if chunk.sample_id == sample_id
    }
    if not questions:
        raise ValueError(f"unknown sample_id: {sample_id}")
    if len(questions) != 1:
        raise ValueError(f"inconsistent questions for sample_id: {sample_id}")
    return questions.pop()


def search_bm25(
    chunks: Sequence[DocumentChunk],
    query: str,
    *,
    top_k: int,
    include_title: bool,
    k1: float,
    b: float,
    epsilon: float,
) -> list[RetrievalResult]:
    return BM25Index(
        chunks,
        include_title=include_title,
        k1=k1,
        b=b,
        epsilon=epsilon,
    ).search(query, top_k=top_k)


def _result_record(
    result: RetrievalResult,
    gold_facts: frozenset[tuple[str, str, int]],
) -> dict[str, Any]:
    covered = sorted(covered_gold_facts(result.chunk, gold_facts))
    return {
        "rank": result.rank,
        "score": round(result.score, 8),
        "chunk_id": result.chunk.chunk_id,
        "sample_id": result.chunk.sample_id,
        "doc_id": result.chunk.doc_id,
        "title": result.chunk.title,
        "sentence_ids": result.chunk.sentence_ids,
        "covered_gold_facts": [
            {
                "sample_id": sample_id,
                "title": title,
                "sentence_id": sentence_id,
            }
            for sample_id, title, sentence_id in covered
        ],
    }


def evaluate_bm25(
    chunks: Sequence[DocumentChunk],
    *,
    corpus_scope: CorpusScope = "global",
    top_ks: Sequence[int] = (1, 5),
    include_title: bool = True,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> dict[str, Any]:
    """Evaluate BM25 with deterministic rankings and fact-level gold labels."""
    if corpus_scope not in {"global", "sample"}:
        raise ValueError("corpus_scope must be 'global' or 'sample'")
    if not chunks:
        raise ValueError("BM25 evaluation requires at least one chunk")
    if not top_ks or any(k <= 0 for k in top_ks):
        raise ValueError("top_ks must contain positive integers")

    ordered_ks = sorted(set(top_ks))
    max_k = max(ordered_ks)
    grouped = group_chunks_by_sample(chunks)
    global_index = (
        BM25Index(
            chunks,
            include_title=include_title,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )
        if corpus_scope == "global"
        else None
    )

    query_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, float]] = []
    total_gold_facts = 0
    for sample_id in sorted(grouped):
        sample_chunks = grouped[sample_id]
        question = question_for_sample(sample_chunks, sample_id)
        gold_facts = gold_facts_for_sample(sample_chunks, sample_id)
        if not gold_facts:
            raise ValueError(f"sample has no represented gold facts: {sample_id}")
        total_gold_facts += len(gold_facts)

        index = global_index or BM25Index(
            sample_chunks,
            include_title=include_title,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )
        results = index.search(question, top_k=max_k)
        metrics = evaluate_query(results, gold_facts, ks=ordered_ks)
        metric_records.append(metrics)
        query_records.append(
            {
                "sample_id": sample_id,
                "question": question,
                "gold_fact_count": len(gold_facts),
                "metrics": {
                    key: round(value, 8)
                    for key, value in sorted(metrics.items())
                },
                "results": [
                    _result_record(result, gold_facts) for result in results
                ],
            }
        )

    aggregate = aggregate_metrics(metric_records)
    return {
        "schema_version": 1,
        "config": {
            "corpus_scope": corpus_scope,
            "top_ks": ordered_ks,
            "include_title": include_title,
            "bm25": {
                "k1": k1,
                "b": b,
                "epsilon": epsilon,
            },
        },
        "dataset": {
            "sample_count": len(grouped),
            "chunk_count": len(chunks),
            "gold_fact_count": total_gold_facts,
        },
        "metrics": {
            key: round(value, 8) for key, value in sorted(aggregate.items())
        },
        "queries": query_records,
    }


def write_evaluation_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.data.chunking import DocumentChunk
from src.evaluation.retrieval import (
    aggregate_metrics,
    covered_gold_facts,
    evaluate_query,
    gold_facts_for_sample,
)
from src.pipelines.bm25 import CorpusScope, group_chunks_by_sample, question_for_sample
from src.retrieval.dense import DenseIndex, TextEncoder, chunk_text
from src.retrieval.types import RetrievalResult


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dense_index(
    chunks: Sequence[DocumentChunk],
    encoder: TextEncoder,
    *,
    batch_size: int,
    include_title: bool,
    normalize: bool,
) -> DenseIndex:
    if not chunks:
        raise ValueError("dense index build requires at least one chunk")
    texts = [chunk_text(chunk, include_title=include_title) for chunk in chunks]
    embeddings = encoder.encode_documents(texts, batch_size=batch_size)
    return DenseIndex(chunks, embeddings, normalize=normalize)


def search_dense(
    index: DenseIndex,
    encoder: TextEncoder,
    query: str,
    *,
    top_k: int,
    batch_size: int,
) -> list[RetrievalResult]:
    if not query.strip():
        return []
    query_vectors = encoder.encode_queries([query], batch_size=batch_size)
    return index.search_vector(query_vectors[0], top_k=top_k)


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


def evaluate_dense(
    index: DenseIndex,
    encoder: TextEncoder,
    *,
    corpus_scope: CorpusScope = "global",
    top_ks: Sequence[int] = (1, 5),
    batch_size: int = 32,
) -> dict[str, Any]:
    if corpus_scope not in {"global", "sample"}:
        raise ValueError("corpus_scope must be 'global' or 'sample'")
    if not top_ks or any(k <= 0 for k in top_ks):
        raise ValueError("top_ks must contain positive integers")

    ordered_ks = sorted(set(top_ks))
    max_k = max(ordered_ks)
    grouped = group_chunks_by_sample(index.chunks)
    sample_ids = sorted(grouped)
    questions = [
        question_for_sample(grouped[sample_id], sample_id)
        for sample_id in sample_ids
    ]
    query_vectors = encoder.encode_queries(questions, batch_size=batch_size)

    query_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, float]] = []
    total_gold_facts = 0
    for sample_id, question, query_vector in zip(
        sample_ids, questions, query_vectors, strict=True
    ):
        sample_chunks = grouped[sample_id]
        gold_facts = gold_facts_for_sample(sample_chunks, sample_id)
        if not gold_facts:
            raise ValueError(f"sample has no represented gold facts: {sample_id}")
        total_gold_facts += len(gold_facts)
        search_index = (
            index if corpus_scope == "global" else index.subset(sample_chunks)
        )
        results = search_index.search_vector(query_vector, top_k=max_k)
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
        "retriever": "dense_faiss",
        "config": {
            "corpus_scope": corpus_scope,
            "top_ks": ordered_ks,
            "batch_size": batch_size,
            "normalize": index.normalize,
            "index_type": "IndexFlatIP",
            "encoder": encoder.metadata,
        },
        "dataset": {
            "sample_count": len(grouped),
            "chunk_count": index.chunk_count,
            "gold_fact_count": total_gold_facts,
        },
        "metrics": {
            key: round(value, 8) for key, value in sorted(aggregate.items())
        },
        "queries": query_records,
    }


def write_json_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def compare_reports(
    bm25_report: dict[str, Any],
    dense_report: dict[str, Any],
) -> dict[str, Any]:
    if bm25_report["dataset"] != dense_report["dataset"]:
        raise ValueError("BM25 and dense reports use different datasets")
    bm25_source = bm25_report.get("source", {})
    dense_source = dense_report.get("source", {})
    if (
        bm25_source.get("chunks_sha256")
        and dense_source.get("chunks_sha256")
        and bm25_source["chunks_sha256"] != dense_source["chunks_sha256"]
    ):
        raise ValueError("BM25 and dense reports use different chunk inputs")

    keys = sorted(set(bm25_report["metrics"]).intersection(dense_report["metrics"]))
    return {
        "schema_version": 1,
        "dataset": dense_report["dataset"],
        "source": dense_source,
        "corpus_scope": dense_report["config"]["corpus_scope"],
        "metrics": {
            key: {
                "bm25": bm25_report["metrics"][key],
                "dense": dense_report["metrics"][key],
                "dense_minus_bm25": round(
                    dense_report["metrics"][key] - bm25_report["metrics"][key],
                    8,
                ),
            }
            for key in keys
        },
    }

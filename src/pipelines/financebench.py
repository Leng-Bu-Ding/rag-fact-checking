from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from src.data.chunking import DocumentChunk
from src.data.financebench import FinanceQuestion
from src.evaluation.financebench import evaluate_finance_rankings
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.types import RetrievalResult

FinanceCorpusScope = Literal["global", "document"]
RankingMap = dict[str, list[RetrievalResult]]


def group_chunks_by_document(
    chunks: Sequence[DocumentChunk],
) -> dict[str, list[DocumentChunk]]:
    grouped: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.page_number is None:
            raise ValueError(f"FinanceBench chunk has no page number: {chunk.chunk_id}")
        if chunk.question or chunk.answer or chunk.contains_supporting_fact:
            raise ValueError("FinanceBench index chunks contain evaluation-only fields")
        grouped[chunk.title].append(chunk)
    return dict(grouped)


def covered_questions(
    questions: Sequence[FinanceQuestion],
    chunks: Sequence[DocumentChunk],
) -> list[FinanceQuestion]:
    documents = set(group_chunks_by_document(chunks))
    return [item for item in questions if item.doc_name in documents]


def build_finance_rankings(
    chunks: Sequence[DocumentChunk],
    questions: Sequence[FinanceQuestion],
    dense_index: DenseIndex,
    query_vectors: Mapping[str, np.ndarray],
    *,
    corpus_scope: FinanceCorpusScope,
    ranking_depth: int,
    bm25_config: Mapping[str, Any],
) -> tuple[dict[str, RankingMap], dict[str, float]]:
    if corpus_scope not in {"global", "document"}:
        raise ValueError("corpus_scope must be 'global' or 'document'")
    if ranking_depth <= 0:
        raise ValueError("ranking_depth must be greater than zero")
    grouped = group_chunks_by_document(chunks)
    if any(item.doc_name not in grouped for item in questions):
        raise ValueError("questions include documents not present in the corpus")

    bm25_global = BM25Index(chunks, **bm25_config) if corpus_scope == "global" else None
    bm25_by_document: dict[str, BM25Index] = {}
    dense_by_document: dict[str, DenseIndex] = {}
    bm25_rankings: RankingMap = {}
    dense_rankings: RankingMap = {}

    bm25_started = time.perf_counter()
    for question in questions:
        index = bm25_global
        if index is None:
            index = bm25_by_document.get(question.doc_name)
            if index is None:
                index = BM25Index(grouped[question.doc_name], **bm25_config)
                bm25_by_document[question.doc_name] = index
        bm25_rankings[question.financebench_id] = index.search(
            question.question, top_k=ranking_depth
        )
    bm25_seconds = time.perf_counter() - bm25_started

    dense_started = time.perf_counter()
    for question in questions:
        index = dense_index
        if corpus_scope == "document":
            index = dense_by_document.get(question.doc_name)
            if index is None:
                index = dense_index.subset(grouped[question.doc_name])
                dense_by_document[question.doc_name] = index
        dense_rankings[question.financebench_id] = index.search_vector(
            query_vectors[question.financebench_id], top_k=ranking_depth
        )
    dense_seconds = time.perf_counter() - dense_started
    count = len(questions)
    return {"bm25": bm25_rankings, "dense": dense_rankings}, {
        "bm25_total_seconds": round(bm25_seconds, 6),
        "bm25_milliseconds_per_query": round(bm25_seconds * 1000 / count, 6),
        "dense_search_total_seconds": round(dense_seconds, 6),
        "dense_search_milliseconds_per_query": round(dense_seconds * 1000 / count, 6),
    }


def fuse_finance_rankings(
    rankings: Mapping[str, RankingMap],
    questions: Sequence[FinanceQuestion],
    *,
    candidate_k: int,
    rrf_k: int,
    top_k: int,
) -> RankingMap:
    return {
        question.financebench_id: reciprocal_rank_fusion(
            {
                name: results[question.financebench_id][:candidate_k]
                for name, results in rankings.items()
            },
            top_k=top_k,
            rrf_k=rrf_k,
        )
        for question in questions
    }


def evaluate_finance_systems(
    questions: Sequence[FinanceQuestion],
    rankings: Mapping[str, RankingMap],
    *,
    top_ks: Sequence[int],
) -> dict[str, Any]:
    return {
        name: evaluate_finance_rankings(questions, values, ks=top_ks)
        for name, values in rankings.items()
    }


def rerank_finance(
    reranker: CrossEncoderReranker,
    questions: Sequence[FinanceQuestion],
    candidates: RankingMap,
    *,
    top_k: int,
) -> tuple[RankingMap, dict[str, float]]:
    started = time.perf_counter()
    output = reranker.rerank_many(
        {item.financebench_id: item.question for item in questions},
        candidates,
        top_k=top_k,
    )
    seconds = time.perf_counter() - started
    return output, {
        "total_seconds": round(seconds, 6),
        "milliseconds_per_query": round(seconds * 1000 / len(questions), 6),
    }

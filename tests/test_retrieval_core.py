from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from src.data.chunking import DocumentChunk
from src.evaluation.retrieval import evaluate_query
from src.pipelines.retrieval_experiment import deterministic_split, split_sha256
from src.retrieval.base import DenseRetriever
from src.retrieval.composite import HybridRetriever
from src.retrieval.dense import DenseIndex
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.types import RetrievalResult


def make_chunk(
    chunk_id: str,
    *,
    sample_id: str = "sample-1",
    title: str = "Document",
    sentence_ids: list[int] | None = None,
    supporting_sentence_ids: list[int] | None = None,
) -> DocumentChunk:
    sentence_ids = sentence_ids or [0]
    supporting = supporting_sentence_ids or []
    return DocumentChunk(
        chunk_id=chunk_id,
        sample_id=sample_id,
        question="question",
        answer="hidden answer",
        doc_id=0,
        title=title,
        text=chunk_id,
        sentence_ids=sentence_ids,
        start_sentence_id=sentence_ids[0],
        end_sentence_id=sentence_ids[-1],
        supporting_sentence_ids=supporting,
        contains_supporting_fact=bool(supporting),
    )


class FakeEncoder:
    @property
    def metadata(self) -> dict[str, Any]:
        return {"dimension": 2}

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        raise AssertionError("document encoding is not used")


class StaticRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.results[:top_k]


class FakeScorer:
    @property
    def metadata(self) -> dict[str, Any]:
        return {"provider": "fake"}

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> np.ndarray:
        return np.asarray(
            [2.0 if "better" in document else 1.0 for _, document in pairs],
            dtype=np.float32,
        )


def result(chunk: DocumentChunk, rank: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(score=score, rank=rank, chunk=chunk)


def test_dense_retriever_uses_common_text_query_interface() -> None:
    chunks = [make_chunk("best"), make_chunk("other")]
    index = DenseIndex(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = DenseRetriever(index, FakeEncoder()).search("question", top_k=1)

    assert results[0].chunk.chunk_id == "best"


def test_hybrid_retriever_fuses_and_deduplicates_results() -> None:
    shared = make_chunk("shared")
    sparse_only = make_chunk("sparse")
    dense_only = make_chunk("dense")
    hybrid = HybridRetriever(
        {
            "bm25": StaticRetriever([result(shared, 1), result(sparse_only, 2)]),
            "dense": StaticRetriever([result(dense_only, 1), result(shared, 2)]),
        },
        candidate_k=2,
        rrf_k=60,
    )

    results = hybrid.search("question", top_k=3)

    assert results[0].chunk.chunk_id == "shared"
    assert len({item.chunk.chunk_id for item in results}) == 3


def test_multihop_metrics_reward_new_facts_not_duplicate_chunks() -> None:
    gold = frozenset(
        {("sample-1", "A", 0), ("sample-1", "B", 0)}
    )
    rankings = [
        result(make_chunk("a1", title="A", supporting_sentence_ids=[0]), 1),
        result(make_chunk("a2", title="A", supporting_sentence_ids=[0]), 2),
        result(make_chunk("b", title="B", supporting_sentence_ids=[0]), 3),
    ]

    metrics = evaluate_query(rankings, gold, ks=(2, 3))

    assert metrics["complete_at_2"] == 0.0
    assert metrics["complete_at_3"] == 1.0
    assert metrics["gold_document_recall_at_2"] == 0.5
    assert metrics["gold_document_recall_at_3"] == 1.0
    assert metrics["fact_ndcg_at_2"] < 1.0
    assert metrics["fact_ndcg_at_3"] < 1.0


def test_cross_encoder_reranker_changes_order_and_reassigns_ranks() -> None:
    candidates = [
        result(make_chunk("ordinary"), 1, score=10.0),
        result(make_chunk("better evidence"), 2, score=1.0),
    ]

    reranked = CrossEncoderReranker(FakeScorer()).rerank(
        "question", candidates, top_k=2
    )

    assert [item.chunk.chunk_id for item in reranked] == [
        "better evidence",
        "ordinary",
    ]
    assert [item.rank for item in reranked] == [1, 2]


def test_cross_encoder_rejects_mismatched_query_keys() -> None:
    reranker = CrossEncoderReranker(FakeScorer())

    with pytest.raises(ValueError, match="identical keys"):
        reranker.rerank_many({"a": "question"}, {"b": []}, top_k=1)


def test_deterministic_split_is_stable_disjoint_and_hashed() -> None:
    ids = [f"sample-{index}" for index in range(10)]

    first = deterministic_split(ids, dev_count=3, seed=42)
    second = deterministic_split(reversed(ids), dev_count=3, seed=42)

    assert first == second
    assert len(first["dev"]) == 3
    assert set(first["dev"]).isdisjoint(first["test"])
    assert len(split_sha256(first)) == 64

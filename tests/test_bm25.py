from __future__ import annotations

import json

import pytest

from src.data.chunk_io import read_chunks_jsonl
from src.data.chunking import DocumentChunk
from src.data.jsonl import write_chunks_jsonl
from src.evaluation.retrieval import (
    aggregate_metrics,
    evaluate_query,
    gold_facts_for_sample,
)
from src.pipelines.bm25 import evaluate_bm25
from src.retrieval.bm25 import BM25Index, tokenize_english
from src.retrieval.types import RetrievalResult


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    sample_id: str = "sample-1",
    question: str = "Where is the quasar evidence?",
    title: str = "Document",
    sentence_ids: list[int] | None = None,
    supporting_sentence_ids: list[int] | None = None,
) -> DocumentChunk:
    sentence_ids = sentence_ids or [0]
    supporting_sentence_ids = supporting_sentence_ids or []
    return DocumentChunk(
        chunk_id=chunk_id,
        sample_id=sample_id,
        question=question,
        answer="answer",
        doc_id=0,
        title=title,
        text=text,
        sentence_ids=sentence_ids,
        start_sentence_id=sentence_ids[0],
        end_sentence_id=sentence_ids[-1],
        supporting_sentence_ids=supporting_sentence_ids,
        contains_supporting_fact=bool(supporting_sentence_ids),
    )


def test_tokenize_english_is_lowercase_and_deterministic() -> None:
    assert tokenize_english("Scott's well-known Film, 1994!") == [
        "scott's",
        "well",
        "known",
        "film",
        "1994",
    ]


def test_bm25_ranks_matching_chunk_and_preserves_provenance() -> None:
    chunks = [
        make_chunk("chunk-c", "ocean tide"),
        make_chunk("chunk-a", "rare quasar evidence", title="Quasar"),
        make_chunk("chunk-b", "forest canopy"),
    ]

    result = BM25Index(chunks).search("quasar", top_k=2)

    assert result[0].chunk.chunk_id == "chunk-a"
    assert result[0].chunk.title == "Quasar"
    assert result[0].rank == 1
    assert result[0].score > result[1].score


def test_bm25_ties_are_broken_by_chunk_id() -> None:
    chunks = [
        make_chunk("chunk-c", "alpha"),
        make_chunk("chunk-a", "beta"),
        make_chunk("chunk-b", "gamma"),
    ]

    result = BM25Index(chunks).search("outofvocabulary", top_k=3)

    assert [item.chunk.chunk_id for item in result] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]
    assert all(item.score == 0.0 for item in result)


def test_bm25_empty_query_returns_no_results() -> None:
    index = BM25Index([make_chunk("chunk-a", "alpha")])

    assert index.search(" \n\t", top_k=1) == []


def test_bm25_rejects_invalid_top_k() -> None:
    index = BM25Index([make_chunk("chunk-a", "alpha")])

    with pytest.raises(ValueError, match="top_k"):
        index.search("alpha", top_k=0)


def test_metrics_deduplicate_overlapping_fact_coverage() -> None:
    gold = frozenset(
        {
            ("sample-1", "Evidence", 1),
            ("sample-1", "Evidence", 2),
        }
    )
    results = [
        RetrievalResult(
            score=3.0,
            rank=1,
            chunk=make_chunk("irrelevant", "none", sample_id="sample-2"),
        ),
        RetrievalResult(
            score=2.0,
            rank=2,
            chunk=make_chunk(
                "first-copy",
                "first",
                title="Evidence",
                sentence_ids=[1],
                supporting_sentence_ids=[1],
            ),
        ),
        RetrievalResult(
            score=1.5,
            rank=3,
            chunk=make_chunk(
                "overlap-copy",
                "first again",
                title="Evidence",
                sentence_ids=[1],
                supporting_sentence_ids=[1],
            ),
        ),
        RetrievalResult(
            score=1.0,
            rank=4,
            chunk=make_chunk(
                "second-fact",
                "second",
                title="Evidence",
                sentence_ids=[2],
                supporting_sentence_ids=[2],
            ),
        ),
    ]

    metrics = evaluate_query(results, gold, ks=(1, 2, 3, 4))

    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_2"] == 1.0
    assert metrics["recall_at_2"] == 0.5
    assert metrics["recall_at_3"] == 0.5
    assert metrics["recall_at_4"] == 1.0
    assert metrics["mrr"] == 0.5


def test_gold_facts_are_scoped_to_the_target_sample() -> None:
    chunks = [
        make_chunk(
            "sample-1-gold",
            "gold",
            title="Same title",
            supporting_sentence_ids=[0],
        ),
        make_chunk(
            "sample-2-gold",
            "other gold",
            sample_id="sample-2",
            title="Same title",
            supporting_sentence_ids=[0],
        ),
    ]

    assert gold_facts_for_sample(chunks, "sample-1") == frozenset(
        {("sample-1", "Same title", 0)}
    )


def test_aggregate_metrics_requires_consistent_keys() -> None:
    with pytest.raises(ValueError, match="same keys"):
        aggregate_metrics([{"mrr": 1.0}, {"hit_at_1": 1.0}])


def test_chunk_jsonl_round_trip(tmp_path) -> None:
    chunks = [make_chunk("chunk-a", "quasar", supporting_sentence_ids=[0])]
    path = tmp_path / "chunks.jsonl"
    write_chunks_jsonl(chunks, path)

    loaded = read_chunks_jsonl(path)

    assert loaded == chunks


def test_evaluation_report_is_deterministic() -> None:
    chunks = [
        make_chunk(
            "sample-1-gold",
            "rare quasar evidence",
            title="Quasar",
            supporting_sentence_ids=[0],
        ),
        make_chunk("sample-1-a", "ocean tide"),
        make_chunk("sample-1-b", "forest canopy"),
    ]

    first = evaluate_bm25(chunks, corpus_scope="sample", top_ks=(1, 3))
    second = evaluate_bm25(chunks, corpus_scope="sample", top_ks=(1, 3))

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["metrics"]["hit_at_1"] == 1.0
    assert first["metrics"]["recall_at_3"] == 1.0

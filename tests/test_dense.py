from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from src.data.chunking import DocumentChunk
from src.pipelines.dense import (
    build_dense_index,
    compare_reports,
    evaluate_dense,
    search_dense,
)
from src.retrieval.dense import DenseIndex, chunk_text


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    sample_id: str = "sample-1",
    question: str = "Where is the quasar evidence?",
    title: str = "Document",
    supporting_sentence_ids: list[int] | None = None,
) -> DocumentChunk:
    supporting = supporting_sentence_ids or []
    return DocumentChunk(
        chunk_id=chunk_id,
        sample_id=sample_id,
        question=question,
        answer="hidden answer",
        doc_id=0,
        title=title,
        text=text,
        sentence_ids=[0],
        start_sentence_id=0,
        end_sentence_id=0,
        supporting_sentence_ids=supporting,
        contains_supporting_fact=bool(supporting),
    )


class FakeEncoder:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "fake",
            "model_name": "test-encoder",
            "revision": "fixed",
            "dimension": 2,
            "device": "cpu",
            "seed": 42,
        }

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        return self._encode(texts)


def test_chunk_text_excludes_answer_and_gold_metadata() -> None:
    chunk = make_chunk("a", "visible text", title="Visible title")

    text = chunk_text(chunk)

    assert text == "Visible title\nvisible text"
    assert "hidden answer" not in text


def test_dense_ranking_and_stable_tie_break() -> None:
    chunks = [
        make_chunk("chunk-c", "c"),
        make_chunk("chunk-a", "a"),
        make_chunk("chunk-b", "b"),
    ]
    embeddings = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    index = DenseIndex(chunks, embeddings)

    results = index.search_vector([1, 0], top_k=3)

    assert [result.chunk.chunk_id for result in results] == [
        "chunk-a",
        "chunk-c",
        "chunk-b",
    ]
    assert [result.rank for result in results] == [1, 2, 3]


def test_dense_rejects_alignment_and_dimension_errors() -> None:
    chunks = [make_chunk("a", "a"), make_chunk("b", "b")]
    with pytest.raises(ValueError, match="row count"):
        DenseIndex(chunks, [[1.0, 0.0]])

    index = DenseIndex(chunks, [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="query dimension"):
        index.search_vector([1.0, 0.0, 0.0])


def test_dense_empty_query_returns_no_results() -> None:
    chunk = make_chunk("a", "doc")
    encoder = FakeEncoder({"Document\ndoc": [1, 0]})
    index = build_dense_index(
        [chunk],
        encoder,
        batch_size=2,
        include_title=True,
        normalize=True,
    )

    assert search_dense(index, encoder, "  ", top_k=1, batch_size=2) == []


def test_faiss_index_round_trip_preserves_chunks(tmp_path) -> None:
    chunks = [
        make_chunk("a", "alpha", supporting_sentence_ids=[0]),
        make_chunk("b", "beta"),
    ]
    index = DenseIndex(chunks, [[1.0, 0.0], [0.0, 1.0]])
    index.save(tmp_path / "index", {"source": {"chunks_sha256": "abc"}})

    loaded, manifest = DenseIndex.load(tmp_path / "index")

    assert loaded.chunks == tuple(chunks)
    assert loaded.dimension == 2
    assert manifest["index"]["chunk_count"] == 2
    assert loaded.search_vector([1, 0], top_k=1)[0].chunk.chunk_id == "a"


def test_dense_evaluation_reuses_fact_metrics() -> None:
    gold = make_chunk(
        "gold",
        "quasar evidence",
        title="Gold",
        supporting_sentence_ids=[0],
    )
    other = make_chunk("other", "ocean")
    encoder = FakeEncoder(
        {
            "Gold\nquasar evidence": [1, 0],
            "Document\nocean": [0, 1],
            "Where is the quasar evidence?": [1, 0],
        }
    )
    index = build_dense_index(
        [gold, other],
        encoder,
        batch_size=2,
        include_title=True,
        normalize=True,
    )

    first = evaluate_dense(index, encoder, corpus_scope="global", top_ks=(1, 2))
    second = evaluate_dense(index, encoder, corpus_scope="sample", top_ks=(1, 2))

    assert first["metrics"]["hit_at_1"] == 1.0
    assert first["metrics"]["recall_at_2"] == 1.0
    assert json.dumps(first, sort_keys=True) == json.dumps(
        evaluate_dense(index, encoder, corpus_scope="global", top_ks=(1, 2)),
        sort_keys=True,
    )
    assert second["metrics"] == first["metrics"]


def test_report_comparison_checks_dataset_and_computes_delta() -> None:
    source = {"chunks_sha256": "same"}
    bm25 = {
        "dataset": {"sample_count": 1, "chunk_count": 2, "gold_fact_count": 1},
        "source": source,
        "metrics": {"hit_at_1": 0.0, "mrr": 0.5},
    }
    dense = {
        "dataset": bm25["dataset"],
        "source": source,
        "config": {"corpus_scope": "global"},
        "metrics": {"hit_at_1": 1.0, "mrr": 1.0},
    }

    comparison = compare_reports(bm25, dense)

    assert comparison["metrics"]["hit_at_1"]["dense_minus_bm25"] == 1.0
    assert comparison["metrics"]["mrr"]["dense_minus_bm25"] == 0.5

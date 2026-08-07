from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.generation.grounded import LocalGroundedGenerator
from src.pipelines.bm25 import question_for_sample
from src.pipelines.dense import search_dense
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex, SentenceTransformerEncoder
from src.retrieval.reranker import (
    CrossEncoderReranker,
    TransformersCrossEncoderScorer,
)
from src.retrieval.hybrid import (
    prioritize_title_mentions,
    reciprocal_rank_fusion,
)
from src.utils.config import load_config


class RAGService:
    """End-to-end local RAG service for the interactive MVP."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        dense_config_path: str | Path | None = None,
    ) -> None:
        root = Path(project_root)
        config_path = dense_config_path or root / "configs" / "dense.yaml"
        config = load_config(config_path)["dense"]
        index_dir = root / config["index_dir"]
        self._dense_index, self._manifest = DenseIndex.load(index_dir)
        self._dense_encoder = SentenceTransformerEncoder(
            str(config["model_name"]),
            revision=config.get("revision"),
            device=str(config["device"]),
            seed=int(config["seed"]),
            cache_folder=config.get("cache_folder"),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        self._bm25_index = BM25Index(self._dense_index.chunks)
        self._batch_size = int(config["batch_size"])
        retrieval_config = load_config(
            root / "configs" / "hotpotqa_retrieval.yaml"
        )["hotpotqa_retrieval"]
        reranker_config = retrieval_config["reranker"]
        self._reranker = CrossEncoderReranker(
            TransformersCrossEncoderScorer(
                str(reranker_config["model_name"]),
                cache_dir=reranker_config.get("cache_dir"),
                device=str(reranker_config["device"]),
                local_files_only=bool(reranker_config["local_files_only"]),
                max_length=int(reranker_config["max_length"]),
            ),
            batch_size=int(reranker_config["batch_size"]),
        )
        self._generator = LocalGroundedGenerator(
            cache_dir=config.get("cache_folder"),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        self._samples = self._collect_samples()

    def _collect_samples(self) -> list[dict[str, str]]:
        seen: set[str] = set()
        samples: list[dict[str, str]] = []
        for chunk in self._dense_index.chunks:
            if chunk.sample_id not in seen:
                seen.add(chunk.sample_id)
                samples.append(
                    {
                        "sample_id": chunk.sample_id,
                        "question": question_for_sample(
                            self._dense_index.chunks, chunk.sample_id
                        ),
                    }
                )
        return samples

    @property
    def samples(self) -> list[dict[str, str]]:
        return list(self._samples)

    @property
    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "chunk_count": self._dense_index.chunk_count,
            "embedding_model": self._dense_encoder.metadata["model_name"],
            "generation_model": self._generator.metadata["model_name"],
            "retrieval": "BM25 + Dense FAISS + Hybrid RRF + Cross-Encoder",
        }

    def ask(self, question: str, *, top_k: int = 5) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")

        started = time.perf_counter()
        candidate_k = max(20, top_k * 4)
        bm25_results = self._bm25_index.search(question, top_k=candidate_k)
        dense_results = search_dense(
            self._dense_index,
            self._dense_encoder,
            question,
            top_k=candidate_k,
            batch_size=self._batch_size,
        )
        retrieval_finished = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_results, "dense": dense_results},
            top_k=candidate_k,
            rrf_k=10,
        )
        reranked = self._reranker.rerank(
            question,
            fused,
            top_k=max(10, top_k * 2),
        )
        reranking_finished = time.perf_counter()
        results = prioritize_title_mentions(question, reranked, top_k=top_k)
        answer = self._generator.generate(question, results)
        finished = time.perf_counter()

        return {
            "question": question,
            "answer": answer,
            "retriever": "hybrid_rrf",
            "evidence": [
                {
                    "citation": index,
                    "rank": result.rank,
                    "score": round(result.score, 8),
                    "chunk_id": result.chunk.chunk_id,
                    "sample_id": result.chunk.sample_id,
                    "title": result.chunk.title,
                    "text": result.chunk.text,
                    "sentence_ids": result.chunk.sentence_ids,
                }
                for index, result in enumerate(results, start=1)
            ],
            "timing_ms": {
                "retrieval": round(
                    (retrieval_finished - started) * 1000, 2
                ),
                "reranking": round(
                    (reranking_finished - retrieval_finished) * 1000, 2
                ),
                "generation": round(
                    (finished - reranking_finished) * 1000, 2
                ),
                "total": round((finished - started) * 1000, 2),
            },
        }

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from src.retrieval.dense import chunk_text
from src.retrieval.types import RetrievalResult


class PairScorer(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> np.ndarray: ...


class TransformersCrossEncoderScorer:
    """CPU-friendly Hugging Face sequence-classification pair scorer."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
        max_length: int = 512,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if device != "cpu":
            raise ValueError("the baseline cross-encoder currently supports CPU only")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._model.eval()
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._local_files_only = local_files_only

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "transformers",
            "model_name": self._model_name,
            "device": self._device,
            "max_length": self._max_length,
            "local_files_only": self._local_files_only,
        }

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not pairs:
            return np.empty(0, dtype=np.float32)

        scores: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self._tokenizer(
                    [query for query, _ in batch],
                    [document for _, document in batch],
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits.detach().cpu().numpy()
                if logits.ndim != 2:
                    raise ValueError("cross-encoder returned invalid logits")
                if logits.shape[1] == 1:
                    values = logits[:, 0]
                elif logits.shape[1] == 2:
                    values = logits[:, 1] - logits[:, 0]
                else:
                    raise ValueError(
                        "cross-encoder must return one score or two class logits"
                    )
                scores.append(values.astype(np.float32, copy=False))
        return np.concatenate(scores)


class CrossEncoderReranker:
    """Rerank retrieved chunks with one score per query-document pair."""

    def __init__(
        self,
        scorer: PairScorer,
        *,
        include_title: bool = True,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self._scorer = scorer
        self._include_title = include_title
        self._batch_size = batch_size

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self._scorer.metadata,
            "include_title": self._include_title,
            "batch_size": self._batch_size,
        }

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        result = self.rerank_many(
            {"query": query},
            {"query": candidates},
            top_k=top_k,
        )
        return result["query"]

    def rerank_many(
        self,
        questions: Mapping[str, str],
        candidates: Mapping[str, Sequence[RetrievalResult]],
        *,
        top_k: int,
    ) -> dict[str, list[RetrievalResult]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if set(questions) != set(candidates):
            raise ValueError("questions and candidates must have identical keys")

        ordered_ids = sorted(questions)
        flat_pairs: list[tuple[str, str]] = []
        spans: dict[str, tuple[int, int]] = {}
        for sample_id in ordered_ids:
            start = len(flat_pairs)
            flat_pairs.extend(
                (
                    questions[sample_id],
                    chunk_text(item.chunk, include_title=self._include_title),
                )
                for item in candidates[sample_id]
            )
            spans[sample_id] = (start, len(flat_pairs))

        flat_scores = self._scorer.score_pairs(
            flat_pairs, batch_size=self._batch_size
        )
        if len(flat_scores) != len(flat_pairs):
            raise ValueError("cross-encoder score count does not match pairs")

        output: dict[str, list[RetrievalResult]] = {}
        for sample_id in ordered_ids:
            start, end = spans[sample_id]
            scored = [
                (float(score), item.chunk)
                for score, item in zip(
                    flat_scores[start:end],
                    candidates[sample_id],
                    strict=True,
                )
            ]
            scored.sort(key=lambda value: (-value[0], value[1].chunk_id))
            output[sample_id] = [
                RetrievalResult(score=score, rank=rank, chunk=chunk)
                for rank, (score, chunk) in enumerate(
                    scored[:top_k], start=1
                )
            ]
        return output

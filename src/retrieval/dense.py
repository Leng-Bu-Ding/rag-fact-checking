from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from src.data.chunk_io import read_chunks_jsonl
from src.data.chunking import DocumentChunk
from src.data.jsonl import write_chunks_jsonl
from src.retrieval.types import RetrievalResult


class TextEncoder(Protocol):
    """Interface used by dense pipelines and lightweight test encoders."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray: ...

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray: ...


def chunk_text(chunk: DocumentChunk, *, include_title: bool = True) -> str:
    """Return retriever-visible text without answer or gold-label leakage."""
    return f"{chunk.title}\n{chunk.text}" if include_title else chunk.text


def _as_float32_matrix(
    values: np.ndarray | Sequence[Sequence[float]],
    *,
    expected_rows: int | None = None,
    name: str = "embeddings",
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{name} row count {matrix.shape[0]} does not match "
            f"chunk count {expected_rows}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _normalized_copy(matrix: np.ndarray, *, name: str) -> np.ndarray:
    normalized = np.array(matrix, dtype=np.float32, order="C", copy=True)
    norms = np.linalg.norm(normalized, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"{name} cannot contain zero vectors")
    faiss.normalize_L2(normalized)
    return normalized


def _chunk_ids_sha256(chunks: Sequence[DocumentChunk]) -> str:
    content = "".join(f"{chunk.chunk_id}\n" for chunk in chunks)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class SentenceTransformerEncoder:
    """Deterministic Sentence Transformers encoder with explicit device use."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        seed: int = 42,
        cache_folder: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")

        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass

        model_kwargs = {"revision": revision} if revision else None
        self._model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs,
            local_files_only=local_files_only,
        )
        self._model_name = model_name
        self._revision = revision
        self._device = device
        self._seed = seed
        self._cache_folder = cache_folder
        self._local_files_only = local_files_only

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "sentence-transformers",
            "model_name": self._model_name,
            "revision": self._revision,
            "dimension": int(self._model.get_embedding_dimension()),
            "device": self._device,
            "seed": self._seed,
            "cache_folder": self._cache_folder,
            "local_files_only": self._local_files_only,
        }

    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not texts:
            return np.empty(
                (0, int(self.metadata["dimension"])), dtype=np.float32
            )
        values = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return _as_float32_matrix(values, expected_rows=len(texts))

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size)

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size)


class DenseIndex:
    """FAISS exact inner-product index aligned with provenance-rich chunks."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray | Sequence[Sequence[float]],
        *,
        normalize: bool = True,
    ) -> None:
        if not chunks:
            raise ValueError("DenseIndex requires at least one chunk")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique")

        matrix = _as_float32_matrix(embeddings, expected_rows=len(chunks))
        self._chunks = tuple(chunks)
        self._normalize = normalize
        self._vectors = (
            _normalized_copy(matrix, name="embeddings")
            if normalize
            else np.array(matrix, copy=True, order="C")
        )
        self._index = faiss.IndexFlatIP(self._vectors.shape[1])
        self._index.add(self._vectors)

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return self._chunks

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1])

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def normalize(self) -> bool:
        return self._normalize

    def subset(self, chunks: Sequence[DocumentChunk]) -> DenseIndex:
        positions = {chunk.chunk_id: index for index, chunk in enumerate(self._chunks)}
        try:
            vectors = np.stack(
                [self._vectors[positions[chunk.chunk_id]] for chunk in chunks]
            )
        except KeyError as error:
            raise ValueError(f"chunk is not present in index: {error.args[0]}") from error
        return DenseIndex(chunks, vectors, normalize=self._normalize)

    def search_vector(
        self,
        query_vector: np.ndarray | Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        vector = _as_float32_matrix(vector, expected_rows=1, name="query vector")
        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"query dimension {vector.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        if self._normalize:
            vector = _normalized_copy(vector, name="query vector")

        # Search all rows so chunk_id can resolve ties at the top-k boundary.
        scores, positions = self._index.search(vector, self.chunk_count)
        candidates = [
            (float(score), self._chunks[int(position)])
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]
        candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))
        candidates = candidates[: min(top_k, self.chunk_count)]
        return [
            RetrievalResult(score=score, rank=rank, chunk=chunk)
            for rank, (score, chunk) in enumerate(candidates, start=1)
        ]

    def save(self, directory: str | Path, manifest: dict[str, Any]) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        write_chunks_jsonl(self._chunks, path / "chunks.jsonl")
        record = {
            **manifest,
            "schema_version": 1,
            "index": {
                "type": "IndexFlatIP",
                "metric": "cosine" if self._normalize else "inner_product",
                "normalize": self._normalize,
                "dimension": self.dimension,
                "chunk_count": self.chunk_count,
                "chunk_ids_sha256": _chunk_ids_sha256(self._chunks),
            },
            "files": {
                "index": "index.faiss",
                "chunks": "chunks.jsonl",
            },
        }
        (path / "manifest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, directory: str | Path) -> tuple[DenseIndex, dict[str, Any]]:
        path = Path(directory)
        manifest = json.loads(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        chunks = read_chunks_jsonl(path / manifest["files"]["chunks"])
        stored = faiss.read_index(str(path / manifest["files"]["index"]))
        index_config = manifest["index"]
        if stored.ntotal != len(chunks):
            raise ValueError("FAISS row count does not match stored chunks")
        if stored.d != int(index_config["dimension"]):
            raise ValueError("FAISS dimension does not match manifest")
        if _chunk_ids_sha256(chunks) != index_config["chunk_ids_sha256"]:
            raise ValueError("stored chunk order does not match manifest")

        vectors = np.empty((stored.ntotal, stored.d), dtype=np.float32)
        stored.reconstruct_n(0, stored.ntotal, vectors)
        instance = cls(
            chunks,
            vectors,
            normalize=bool(index_config["normalize"]),
        )
        return instance, manifest

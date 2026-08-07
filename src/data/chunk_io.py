from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.chunking import DocumentChunk


def _chunk_from_record(record: dict[str, Any]) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(record["chunk_id"]),
        sample_id=str(record["sample_id"]),
        question=str(record["question"]),
        answer=str(record["answer"]),
        doc_id=int(record["doc_id"]),
        title=str(record["title"]),
        text=str(record["text"]),
        sentence_ids=[int(value) for value in record["sentence_ids"]],
        start_sentence_id=int(record["start_sentence_id"]),
        end_sentence_id=int(record["end_sentence_id"]),
        supporting_sentence_ids=[
            int(value) for value in record["supporting_sentence_ids"]
        ],
        contains_supporting_fact=bool(record["contains_supporting_fact"]),
    )


def read_chunks_jsonl(input_path: str | Path) -> list[DocumentChunk]:
    """Read chunks from UTF-8 JSONL and report the failing source line."""
    path = Path(input_path)
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            try:
                record: dict[str, Any] = json.loads(line)
                chunks.append(_chunk_from_record(record))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid chunk record at {path}:{line_number}: {error}"
                ) from error
    return chunks

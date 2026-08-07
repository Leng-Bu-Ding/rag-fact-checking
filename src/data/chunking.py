from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.data.load_hotpotqa import UnifiedSample

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    sample_id: str
    question: str
    answer: str
    doc_id: int
    title: str
    text: str
    sentence_ids: list[int]
    start_sentence_id: int
    end_sentence_id: int
    supporting_sentence_ids: list[int]
    contains_supporting_fact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_text(text: str) -> str:
    """Collapse whitespace while leaving words and punctuation unchanged."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _joined_length(sentences: list[tuple[int, str]]) -> int:
    return len(" ".join(text for _, text in sentences))


def _next_start(
    sentences: list[tuple[int, str]],
    current_start: int,
    current_end: int,
    overlap_chars: int,
) -> int:
    if overlap_chars <= 0 or current_end - current_start <= 1:
        return current_end

    next_start = current_end
    for candidate in range(current_end - 1, current_start, -1):
        if _joined_length(sentences[candidate:current_end]) <= overlap_chars:
            next_start = candidate
        else:
            break
    return next_start if next_start > current_start else current_end


def _sentence_windows(
    sentences: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable[list[tuple[int, str]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    start = 0
    while start < len(sentences):
        end = start
        while end < len(sentences):
            candidate = sentences[start : end + 1]
            if end > start and _joined_length(candidate) > chunk_size:
                break
            end += 1

        yield sentences[start:end]
        start = _next_start(sentences, start, end, chunk_overlap)


def _make_chunk_id(
    sample_id: str,
    doc_id: int,
    sentence_ids: list[int],
    text: str,
) -> str:
    identity = f"{sample_id}|{doc_id}|{sentence_ids}|{text}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return (
        f"{sample_id}_d{doc_id}_s{sentence_ids[0]}-"
        f"{sentence_ids[-1]}_{digest}"
    )


def chunk_sample(
    sample: UnifiedSample,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_text_length: int = 1,
) -> list[DocumentChunk]:
    if min_text_length < 1:
        raise ValueError("min_text_length must be at least one")

    supporting_by_title: dict[str, set[int]] = {}
    for fact in sample.supporting_facts:
        title = str(fact["title"])
        supporting_by_title.setdefault(title, set()).add(int(fact["sent_id"]))

    chunks: list[DocumentChunk] = []
    for document in sample.documents:
        indexed_sentences = [
            (sentence_id, cleaned)
            for sentence_id, sentence in enumerate(document.get("sentences", []))
            if (cleaned := clean_text(str(sentence)))
        ]
        if not indexed_sentences:
            continue

        title = str(document.get("title", ""))
        supporting_ids = supporting_by_title.get(title, set())
        for window in _sentence_windows(
            indexed_sentences,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            text = " ".join(sentence for _, sentence in window)
            if len(text) < min_text_length:
                continue

            sentence_ids = [sentence_id for sentence_id, _ in window]
            matched_supporting_ids = sorted(supporting_ids.intersection(sentence_ids))
            doc_id = int(document["doc_id"])
            chunks.append(
                DocumentChunk(
                    chunk_id=_make_chunk_id(
                        sample.sample_id,
                        doc_id,
                        sentence_ids,
                        text,
                    ),
                    sample_id=sample.sample_id,
                    question=sample.question,
                    answer=sample.answer,
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    sentence_ids=sentence_ids,
                    start_sentence_id=sentence_ids[0],
                    end_sentence_id=sentence_ids[-1],
                    supporting_sentence_ids=matched_supporting_ids,
                    contains_supporting_fact=bool(matched_supporting_ids),
                )
            )
    return chunks


def chunk_samples(
    samples: Iterable[UnifiedSample],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_text_length: int = 1,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for sample in samples:
        chunks.extend(
            chunk_sample(
                sample,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_text_length=min_text_length,
            )
        )
    return chunks

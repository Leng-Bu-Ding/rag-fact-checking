from __future__ import annotations

import json

import pytest

from src.data.chunking import chunk_sample, clean_text
from src.data.jsonl import write_chunks_jsonl
from src.data.load_hotpotqa import UnifiedSample, to_unified_sample


def make_sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="sample-1",
        question="Which fact is supported?",
        answer="Alpha",
        documents=[
            {
                "doc_id": 0,
                "title": "Evidence",
                "sentences": [
                    " Alpha   is the first fact. ",
                    "Beta is another fact.",
                    "Gamma closes the document.",
                ],
                "text": "",
            }
        ],
        supporting_facts=[{"title": "Evidence", "sent_id": 1}],
    )


def test_clean_text_only_normalizes_whitespace() -> None:
    assert clean_text("  Alpha\n\t beta.  ") == "Alpha beta."


def test_normalization_preserves_sentence_ids() -> None:
    sample = to_unified_sample(
        {
            "id": "raw-1",
            "question": " Question? ",
            "answer": " Answer ",
            "context": {
                "title": ["Doc"],
                "sentences": [["First.", "Second."]],
            },
            "supporting_facts": {"title": ["Doc"], "sent_id": [1]},
        }
    )

    assert sample.question == "Question?"
    assert sample.documents[0]["sentences"][1] == "Second."
    assert sample.supporting_facts == [{"title": "Doc", "sent_id": 1}]


def test_chunking_preserves_provenance_and_support_labels() -> None:
    chunks = chunk_sample(make_sample(), chunk_size=55, chunk_overlap=25)

    assert len(chunks) == 2
    assert chunks[0].sentence_ids == [0, 1]
    assert chunks[1].sentence_ids == [1, 2]
    assert chunks[0].supporting_sentence_ids == [1]
    assert chunks[1].supporting_sentence_ids == [1]
    assert all(chunk.contains_supporting_fact for chunk in chunks)
    assert all(chunk.sample_id == "sample-1" for chunk in chunks)
    assert all(chunk.doc_id == 0 for chunk in chunks)


def test_chunk_ids_are_deterministic() -> None:
    first = chunk_sample(make_sample(), chunk_size=55, chunk_overlap=25)
    second = chunk_sample(make_sample(), chunk_size=55, chunk_overlap=25)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_empty_sentences_are_ignored() -> None:
    sample = make_sample()
    sample.documents[0]["sentences"] = [" ", "\n\t"]

    assert chunk_sample(sample) == []


def test_long_sentence_remains_whole() -> None:
    sample = make_sample()
    sample.documents[0]["sentences"] = ["x" * 80]

    chunks = chunk_sample(sample, chunk_size=20, chunk_overlap=0)

    assert len(chunks) == 1
    assert len(chunks[0].text) == 80
    assert chunks[0].sentence_ids == [0]


def test_invalid_chunk_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_sample(make_sample(), chunk_size=64, chunk_overlap=64)


def test_jsonl_export_is_utf8_and_deterministic(tmp_path) -> None:
    chunks = chunk_sample(make_sample(), chunk_size=55, chunk_overlap=25)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    assert write_chunks_jsonl(chunks, first_path) == len(chunks)
    write_chunks_jsonl(chunks, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    records = [
        json.loads(line)
        for line in first_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["chunk_id"] == chunks[0].chunk_id

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class UnifiedSample:
    sample_id: str
    question: str
    answer: str
    documents: list[dict[str, Any]]
    supporting_facts: list[dict[str, Any]]


def _normalize_context(context: dict[str, list[Any]]) -> list[dict[str, Any]]:
    titles = context.get("title", [])
    sentences = context.get("sentences", [])

    documents: list[dict[str, Any]] = []
    for doc_id, (title, sentence_list) in enumerate(zip(titles, sentences)):
        text = " ".join(sentence_list).strip()
        documents.append(
            {
                "doc_id": doc_id,
                "title": title,
                "sentences": sentence_list,
                "text": text,
            }
        )
    return documents


def _normalize_supporting_facts(supporting_facts: dict[str, list[Any]]) -> list[dict[str, Any]]:
    titles = supporting_facts.get("title", [])
    sent_ids = supporting_facts.get("sent_id", [])

    return [
        {
            "title": title,
            "sent_id": sent_id,
        }
        for title, sent_id in zip(titles, sent_ids)
    ]


def to_unified_sample(example: dict[str, Any]) -> UnifiedSample:
    return UnifiedSample(
        sample_id=example.get("id", ""),
        question=example.get("question", "").strip(),
        answer=example.get("answer", "").strip(),
        documents=_normalize_context(example.get("context", {})),
        supporting_facts=_normalize_supporting_facts(example.get("supporting_facts", {})),
    )


def load_hotpotqa_samples(
    sample_size: int = 5,
    subset: str = "distractor",
    split: str = "validation",
    cache_dir: str | None = None,
) -> list[UnifiedSample]:
    dataset = load_dataset("hotpotqa/hotpot_qa", subset, split=split, cache_dir=cache_dir)
    selected = dataset.select(range(min(sample_size, len(dataset))))
    return [to_unified_sample(example) for example in selected]

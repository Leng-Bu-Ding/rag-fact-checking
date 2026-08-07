from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from src.data.chunking import DocumentChunk
from src.retrieval.types import RetrievalResult

GoldFact = tuple[str, str, int]


def gold_facts_for_sample(
    chunks: Iterable[DocumentChunk],
    sample_id: str,
) -> frozenset[GoldFact]:
    """Return unique gold facts represented by chunks for one sample."""
    return frozenset(
        (sample_id, chunk.title, sentence_id)
        for chunk in chunks
        if chunk.sample_id == sample_id
        for sentence_id in chunk.supporting_sentence_ids
    )


def covered_gold_facts(
    chunk: DocumentChunk,
    gold_facts: frozenset[GoldFact],
) -> frozenset[GoldFact]:
    """Return target gold facts covered by a retrieved chunk."""
    candidates = {
        (chunk.sample_id, chunk.title, sentence_id)
        for sentence_id in chunk.supporting_sentence_ids
    }
    return frozenset(candidates.intersection(gold_facts))


def evaluate_query(
    results: Sequence[RetrievalResult],
    gold_facts: frozenset[GoldFact],
    *,
    ks: Sequence[int] = (1, 5),
) -> dict[str, float]:
    """Compute fact- and document-level multi-hop retrieval metrics."""
    if not gold_facts:
        raise ValueError("retrieval evaluation requires at least one gold fact")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")

    ordered_ks = sorted(set(ks))
    metrics: dict[str, float] = {}
    gold_documents = {(sample_id, title) for sample_id, title, _ in gold_facts}
    for k in ordered_ks:
        covered: set[GoldFact] = set()
        covered_documents: set[tuple[str, str]] = set()
        novelty_gains: list[float] = []
        for result in results[:k]:
            newly_covered = covered_gold_facts(result.chunk, gold_facts).difference(
                covered
            )
            novelty_gains.append(float(bool(newly_covered)))
            covered.update(newly_covered)
            covered_documents.update(
                (sample_id, title) for sample_id, title, _ in newly_covered
            )
        metrics[f"hit_at_{k}"] = float(bool(covered))
        metrics[f"recall_at_{k}"] = len(covered) / len(gold_facts)
        metrics[f"complete_at_{k}"] = float(covered == set(gold_facts))
        metrics[f"gold_document_recall_at_{k}"] = (
            len(covered_documents) / len(gold_documents)
        )
        dcg = sum(
            gain / math.log2(rank + 1)
            for rank, gain in enumerate(novelty_gains, start=1)
        )
        ideal_hits = min(k, len(gold_facts))
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )
        metrics[f"fact_ndcg_at_{k}"] = dcg / ideal_dcg

    reciprocal_rank = 0.0
    for result in results:
        if covered_gold_facts(result.chunk, gold_facts):
            reciprocal_rank = 1.0 / result.rank
            break
    metrics["mrr"] = reciprocal_rank
    return metrics


def aggregate_metrics(
    query_metrics: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Macro-average per-query metrics with stable key ordering."""
    if not query_metrics:
        raise ValueError("cannot aggregate an empty metrics collection")

    expected_keys = set(query_metrics[0])
    if any(set(metrics) != expected_keys for metrics in query_metrics):
        raise ValueError("all query metrics must have the same keys")

    return {
        key: sum(metrics[key] for metrics in query_metrics) / len(query_metrics)
        for key in sorted(expected_keys)
    }

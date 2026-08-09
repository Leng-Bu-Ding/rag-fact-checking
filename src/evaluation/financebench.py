from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.financebench import FinanceQuestion
from src.evaluation.retrieval import aggregate_metrics
from src.retrieval.types import RetrievalResult

GoldPage = tuple[str, int]


def gold_pages(question: FinanceQuestion) -> frozenset[GoldPage]:
    return frozenset((item.doc_name, item.page_number) for item in question.evidence)


def evaluate_finance_query(
    results: Sequence[RetrievalResult],
    question: FinanceQuestion,
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    pages = gold_pages(question)
    if not pages:
        raise ValueError("FinanceBench evaluation requires gold evidence pages")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    gold_documents = {doc_name for doc_name, _ in pages}
    metrics: dict[str, float] = {}
    for k in sorted(set(ks)):
        top = results[:k]
        retrieved_documents = {item.chunk.title for item in top}
        retrieved_pages = {
            (item.chunk.title, item.chunk.page_number)
            for item in top
            if item.chunk.page_number is not None
        }
        covered = pages.intersection(retrieved_pages)
        relevance = [
            float((item.chunk.title, item.chunk.page_number) in pages)
            for item in top
        ]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(relevance, start=1))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(pages)) + 1))
        metrics[f"document_hit_at_{k}"] = float(bool(gold_documents.intersection(retrieved_documents)))
        metrics[f"evidence_page_hit_at_{k}"] = float(bool(covered))
        metrics[f"evidence_page_recall_at_{k}"] = len(covered) / len(pages)
        metrics[f"ndcg_at_{k}"] = dcg / ideal
    metrics["mrr"] = next(
        (
            1.0 / rank
            for rank, item in enumerate(results, start=1)
            if (item.chunk.title, item.chunk.page_number) in pages
        ),
        0.0,
    )
    return metrics


def evaluate_finance_rankings(
    questions: Sequence[FinanceQuestion],
    rankings: Mapping[str, Sequence[RetrievalResult]],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, float]] = []
    for question in sorted(questions, key=lambda item: item.financebench_id):
        results = rankings.get(question.financebench_id)
        if results is None:
            raise ValueError(f"missing ranking for {question.financebench_id}")
        query_metrics = evaluate_finance_query(results, question, ks=ks)
        metrics.append(query_metrics)
        pages = gold_pages(question)
        records.append(
            {
                "financebench_id": question.financebench_id,
                "company": question.company,
                "doc_name": question.doc_name,
                "question_type": question.question_type,
                "question": question.question,
                "gold_pages": [
                    {"doc_name": name, "page_number": page}
                    for name, page in sorted(pages)
                ],
                "metrics": {key: round(value, 8) for key, value in sorted(query_metrics.items())},
                "results": [
                    {
                        "rank": item.rank,
                        "score": round(item.score, 8),
                        "chunk_id": item.chunk.chunk_id,
                        "doc_name": item.chunk.title,
                        "page_number": item.chunk.page_number,
                        "is_gold_page": (item.chunk.title, item.chunk.page_number) in pages,
                    }
                    for item in results[: max(ks)]
                ],
            }
        )
    return {
        "question_count": len(records),
        "metrics": {key: round(value, 8) for key, value in aggregate_metrics(metrics).items()},
        "questions": records,
    }


def summarize_finance_failures(
    evaluation: Mapping[str, Any], *, k: int = 10
) -> dict[str, Any]:
    """Separate successful page retrieval from document and page failures."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    records = evaluation.get("questions")
    if not isinstance(records, list):
        raise ValueError("evaluation must contain a questions list")

    outcome_counts: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        metrics = record["metrics"]
        if metrics[f"evidence_page_hit_at_{k}"]:
            outcome = "gold_page_retrieved"
        elif metrics[f"document_hit_at_{k}"]:
            outcome = "correct_document_wrong_page"
        else:
            outcome = "correct_document_not_retrieved"
        question_type = str(record["question_type"])
        outcome_counts[outcome] += 1
        by_type[question_type][outcome] += 1

    total = len(records)
    return {
        "k": k,
        "question_count": total,
        "outcomes": {
            name: {
                "count": count,
                "rate": round(count / total, 8) if total else 0.0,
            }
            for name, count in sorted(outcome_counts.items())
        },
        "by_question_type": {
            question_type: {
                "question_count": sum(counts.values()),
                "outcomes": dict(sorted(counts.items())),
            }
            for question_type, counts in sorted(by_type.items())
        },
    }

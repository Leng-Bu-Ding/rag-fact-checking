"""Evaluation utilities for retrieval and grounded generation."""

from src.evaluation.retrieval import (
    GoldFact,
    aggregate_metrics,
    covered_gold_facts,
    evaluate_query,
    gold_facts_for_sample,
)

__all__ = [
    "GoldFact",
    "aggregate_metrics",
    "covered_gold_facts",
    "evaluate_query",
    "gold_facts_for_sample",
]

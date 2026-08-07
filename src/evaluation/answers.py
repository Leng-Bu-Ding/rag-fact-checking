from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

from src.evaluation.retrieval import GoldFact, covered_gold_facts
from src.retrieval.types import RetrievalResult

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_ABSTENTION_MARKERS = ("insufficient", "cannot answer", "not enough evidence")


def normalize_answer(text: str) -> str:
    lowered = text.casefold()
    without_citations = _CITATION_RE.sub(" ", lowered)
    without_punctuation = "".join(
        " " if character in string.punctuation else character
        for character in without_citations
    )
    without_articles = _ARTICLE_RE.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(
    prediction: str,
    reference: str,
    evidence: Sequence[RetrievalResult],
    gold_facts: frozenset[GoldFact],
) -> dict[str, float]:
    citation_ids = [int(value) for value in _CITATION_RE.findall(prediction)]
    valid_ids = sorted({value for value in citation_ids if 1 <= value <= len(evidence)})
    cited_results = [evidence[value - 1] for value in valid_ids]
    covered = set()
    supporting_citations = 0
    for result in cited_results:
        result_coverage = covered_gold_facts(result.chunk, gold_facts)
        covered.update(result_coverage)
        supporting_citations += int(bool(result_coverage))

    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    lowered = prediction.casefold()
    return {
        "exact_match": float(normalized_prediction == normalized_reference),
        "token_f1": token_f1(prediction, reference),
        "has_citation": float(bool(citation_ids)),
        "citation_validity": (
            sum(1 for value in citation_ids if 1 <= value <= len(evidence))
            / len(citation_ids)
            if citation_ids
            else 0.0
        ),
        "citation_precision": (
            supporting_citations / len(cited_results) if cited_results else 0.0
        ),
        "citation_recall": len(covered) / len(gold_facts),
        "abstained": float(any(marker in lowered for marker in _ABSTENTION_MARKERS)),
    }

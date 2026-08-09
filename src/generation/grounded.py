from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from src.retrieval.types import RetrievalResult

_CITATION_RE = re.compile(r"\[(\d+)\]")
_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with",
}


def evidence_is_insufficient(
    question: str,
    results: Sequence[RetrievalResult],
) -> bool:
    """Reject clear out-of-domain queries with no lexical evidence connection."""
    if not results:
        return True
    question_terms = {
        token
        for token in _CONTENT_TOKEN_RE.findall(question.casefold())
        if len(token) >= 3 and token not in _QUESTION_STOPWORDS
    }
    if not question_terms:
        return False
    evidence_text = " ".join(
        f"{result.chunk.title} {result.chunk.text}" for result in results
    ).casefold()
    evidence_terms = set(_CONTENT_TOKEN_RE.findall(evidence_text))
    return question_terms.isdisjoint(evidence_terms)


def ensure_valid_citations(
    answer: str,
    question: str,
    results: Sequence[RetrievalResult],
) -> str:
    """Remove invalid citation IDs without inventing model attribution."""
    valid = {
        int(match)
        for match in _CITATION_RE.findall(answer)
        if 1 <= int(match) <= len(results)
    }
    cleaned = _CITATION_RE.sub(
        lambda match: match.group(0)
        if int(match.group(1)) in valid
        else "",
        answer,
    ).strip()
    return cleaned


class LocalGroundedGenerator:
    """Small local FLAN-T5 generator constrained by retrieved evidence."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        *,
        cache_dir: str | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        if device != "cpu":
            raise ValueError("the MVP generator currently supports CPU only")

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._model.eval()
        self._metadata = {
            "provider": "transformers",
            "model_name": model_name,
            "device": device,
            "local_files_only": local_files_only,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def generate(
        self,
        question: str,
        results: Sequence[RetrievalResult],
        *,
        max_evidence: int = 3,
        max_new_tokens: int = 80,
    ) -> str:
        evidence = list(results[:max_evidence])
        if evidence_is_insufficient(question, evidence):
            return "The retrieved evidence is insufficient to answer this question."
        context = "\n".join(
            f"Evidence {index} - {result.chunk.title}: {result.chunk.text}"
            for index, result in enumerate(evidence, start=1)
        )
        prompt = (
            "Answer the question using only the evidence. Write one concise, "
            "complete grammatical sentence and cite supporting evidence with "
            "bracketed IDs such as [1]. Cite only evidence that directly supports "
            "the answer. "
            "If the evidence is insufficient, say that it is insufficient.\n\n"
            f"Evidence:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        output = self._model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=2,
            early_stopping=True,
        )
        answer = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if not answer:
            answer = "The retrieved evidence is insufficient."
        return ensure_valid_citations(answer, question, evidence)

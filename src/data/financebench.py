from __future__ import annotations

import hashlib
import json
import re
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, urlparse

from src.data.chunking import DocumentChunk, clean_text

DATASET_NAME = "PatronusAI/financebench"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"


@dataclass(frozen=True)
class EvidenceRef:
    """Gold evidence used for evaluation only, never for corpus construction."""

    doc_name: str
    source_page_index: int
    page_number: int
    evidence_text: str


@dataclass(frozen=True)
class FinanceQuestion:
    financebench_id: str
    company: str
    doc_name: str
    question_type: str
    question: str
    answer: str
    justification: str
    doc_type: str
    doc_period: str
    doc_link: str
    evidence: tuple[EvidenceRef, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [asdict(item) for item in self.evidence]
        return value


@dataclass(frozen=True)
class FinanceDocument:
    doc_name: str
    company: str
    doc_type: str
    doc_period: str
    source_url: str
    local_path: str
    original_source_url: str | None = None
    status: str = "pending"
    sha256: str | None = None
    byte_count: int | None = None
    page_count: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_text(record: dict[str, Any], field: str) -> str:
    value = str(record.get(field, "")).strip()
    if not value:
        raise ValueError(f"FinanceBench record is missing {field}")
    return value


def _evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value]
    if isinstance(value, dict):
        keys = tuple(value)
        lengths = {len(value[key]) for key in keys}
        if len(lengths) != 1:
            raise ValueError("FinanceBench evidence columns have unequal lengths")
        return [
            {key: value[key][index] for key in keys}
            for index in range(next(iter(lengths), 0))
        ]
    raise ValueError("FinanceBench evidence must be a list or column mapping")


def finance_question_from_record(record: dict[str, Any]) -> FinanceQuestion:
    evidence_items = _evidence_items(record.get("evidence", []))
    evidence = tuple(
        EvidenceRef(
            doc_name=_required_text(item, "doc_name"),
            source_page_index=int(item["evidence_page_num"]),
            page_number=int(item["evidence_page_num"]) + 1,
            evidence_text=_required_text(item, "evidence_text"),
        )
        for item in evidence_items
    )
    if not evidence:
        raise ValueError("FinanceBench question has no gold evidence")
    if any(item.source_page_index < 0 for item in evidence):
        raise ValueError("FinanceBench source page indexes cannot be negative")
    return FinanceQuestion(
        financebench_id=_required_text(record, "financebench_id"),
        company=_required_text(record, "company"),
        doc_name=_required_text(record, "doc_name"),
        question_type=_required_text(record, "question_type"),
        question=_required_text(record, "question"),
        answer=_required_text(record, "answer"),
        justification=str(record.get("justification", "")).strip(),
        doc_type=_required_text(record, "doc_type"),
        doc_period=str(record.get("doc_period", "")).strip(),
        doc_link=_required_text(record, "doc_link"),
        evidence=evidence,
    )


class FinanceBenchAdapter:
    """Convert public FinanceBench rows into validated, deterministic records."""

    def __init__(self, questions: Sequence[FinanceQuestion]) -> None:
        if not questions:
            raise ValueError("FinanceBenchAdapter requires at least one question")
        ids = [item.financebench_id for item in questions]
        if len(set(ids)) != len(ids):
            raise ValueError("FinanceBench question IDs must be unique")
        self._questions = tuple(sorted(questions, key=lambda item: item.financebench_id))

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> FinanceBenchAdapter:
        return cls([finance_question_from_record(record) for record in records])

    @classmethod
    def from_huggingface(
        cls,
        *,
        cache_dir: str | None = None,
    ) -> FinanceBenchAdapter:
        from datasets import load_dataset

        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            cache_dir=cache_dir,
        )
        return cls.from_records(dict(row) for row in dataset)

    @property
    def questions(self) -> tuple[FinanceQuestion, ...]:
        return self._questions

    def documents(self, *, relative_directory: str = "data/raw/financebench/pdfs") -> list[FinanceDocument]:
        records: dict[str, FinanceDocument] = {}
        for question in self._questions:
            candidate = FinanceDocument(
                doc_name=question.doc_name,
                company=question.company,
                doc_type=question.doc_type,
                doc_period=question.doc_period,
                source_url=question.doc_link,
                local_path=f"{relative_directory}/{safe_pdf_filename(question.doc_name)}",
            )
            previous = records.get(question.doc_name)
            if previous and (
                previous.source_url != candidate.source_url
                or previous.company != candidate.company
            ):
                raise ValueError(f"conflicting metadata for document {question.doc_name}")
            records[question.doc_name] = candidate
        return [records[name] for name in sorted(records)]


def safe_pdf_filename(doc_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_name.strip()).strip("._")
    if not stem:
        stem = hashlib.sha256(doc_name.encode("utf-8")).hexdigest()[:16]
    return stem if stem.casefold().endswith(".pdf") else f"{stem}.pdf"


def write_questions_jsonl(
    questions: Iterable[FinanceQuestion], output_path: str | Path
) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for question in questions:
            output.write(json.dumps(question.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_questions_jsonl(input_path: str | Path) -> list[FinanceQuestion]:
    questions: list[FinanceQuestion] = []
    with Path(input_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
                record["evidence"] = [
                    {
                        "doc_name": item["doc_name"],
                        "evidence_page_num": item["source_page_index"],
                        "evidence_text": item["evidence_text"],
                    }
                    for item in record["evidence"]
                ]
                questions.append(finance_question_from_record(record))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid FinanceBench record at {input_path}:{line_number}: {error}") from error
    return questions


def write_document_manifest(
    documents: Iterable[FinanceDocument], output_path: str | Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "documents": [item.to_dict() for item in documents],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_document_manifest(input_path: str | Path) -> list[FinanceDocument]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return [FinanceDocument(**record) for record in payload["documents"]]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_download_url(source_url: str) -> str:
    """Unwrap known vendor viewer URLs while preserving direct public links."""
    parsed = urlparse(source_url)
    target = parse_qs(parsed.query).get("pdfTarget")
    if parsed.netloc.casefold().endswith("adobe.com") and target:
        padded = target[0] + "=" * (-len(target[0]) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8")
    return source_url


def _download_document(document: FinanceDocument, project_root: Path, timeout: int) -> FinanceDocument:
    import requests

    output = project_root / document.local_path
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if output.exists():
        content = output.read_bytes()
        if content.startswith(b"%PDF"):
            return replace(
                document,
                status=("parsed" if document.status == "parsed" else "downloaded"),
                sha256=hashlib.sha256(content).hexdigest(),
                byte_count=len(content),
                error=None,
            )
    try:
        response = requests.get(
            resolve_download_url(document.source_url),
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; EvalRAG-FinanceBench/1.0)",
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.5",
            },
            timeout=(20, timeout),
            allow_redirects=True,
        )
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF"):
            raise ValueError("downloaded content is not a PDF")
        temporary.write_bytes(content)
        temporary.replace(output)
        return replace(
            document,
            status="downloaded",
            sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
            error=None,
        )
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        return replace(document, status="failed", error=f"{type(error).__name__}: {error}")


def download_documents(
    documents: Sequence[FinanceDocument],
    project_root: str | Path,
    *,
    workers: int = 6,
    timeout: int = 60,
) -> list[FinanceDocument]:
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    root = Path(project_root)
    results: dict[str, FinanceDocument] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_document, item, root, timeout): item.doc_name
            for item in documents
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[item.doc_name] for item in documents]


def apply_alternate_urls(
    documents: Sequence[FinanceDocument],
    alternate_urls: dict[str, str],
) -> list[FinanceDocument]:
    unknown = set(alternate_urls).difference(item.doc_name for item in documents)
    if unknown:
        raise ValueError(f"alternate URLs reference unknown documents: {sorted(unknown)}")
    return [
        replace(
            item,
            source_url=alternate_urls[item.doc_name],
            original_source_url=item.original_source_url or item.source_url,
            status="failed",
            error="original source unavailable; alternate source pending",
        )
        if item.doc_name in alternate_urls and item.status not in {"downloaded", "parsed"}
        else item
        for item in documents
    ]


def _word_windows(words: Sequence[str], size: int, overlap: int) -> Iterable[tuple[int, int, str]]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("word chunk size must be positive and overlap smaller than size")
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield start, end, " ".join(words[start:end])
        if end == len(words):
            break
        start = end - overlap


def chunk_finance_pdf(
    document: FinanceDocument,
    project_root: str | Path,
    *,
    doc_id: int,
    chunk_words: int = 350,
    overlap_words: int = 50,
) -> tuple[list[DocumentChunk], int]:
    """Parse a PDF into page-bounded chunks; no gold evidence is accepted here."""
    import pymupdf

    path = Path(project_root) / document.local_path
    chunks: list[DocumentChunk] = []
    with pymupdf.open(path) as pdf:
        page_count = len(pdf)
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            words = clean_text(page.get_text("text")).split()
            for window_id, (start, end, text) in enumerate(
                _word_windows(words, chunk_words, overlap_words)
            ):
                identity = f"{document.doc_name}|{page_number}|{start}|{end}|{text}"
                digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"finance_d{doc_id}_p{page_number}_w{window_id}_{digest}",
                        sample_id=document.doc_name,
                        question="",
                        answer="",
                        doc_id=doc_id,
                        title=document.doc_name,
                        text=text,
                        sentence_ids=[window_id],
                        start_sentence_id=window_id,
                        end_sentence_id=window_id,
                        supporting_sentence_ids=[],
                        contains_supporting_fact=False,
                        page_number=page_number,
                        company=document.company,
                        document_type=document.doc_type,
                        document_period=document.doc_period,
                        source_url=document.source_url,
                    )
                )
    return chunks, page_count


def build_finance_chunks(
    documents: Sequence[FinanceDocument],
    project_root: str | Path,
    *,
    chunk_words: int = 350,
    overlap_words: int = 50,
) -> tuple[list[DocumentChunk], list[FinanceDocument]]:
    chunks: list[DocumentChunk] = []
    updated: list[FinanceDocument] = []
    for doc_id, document in enumerate(documents):
        if document.status not in {"downloaded", "parsed"}:
            updated.append(document)
            continue
        try:
            document_chunks, page_count = chunk_finance_pdf(
                document,
                project_root,
                doc_id=doc_id,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
            chunks.extend(document_chunks)
            updated.append(replace(document, status="parsed", page_count=page_count, error=None))
        except Exception as error:
            updated.append(replace(document, status="parse_failed", error=f"{type(error).__name__}: {error}"))
    return chunks, updated

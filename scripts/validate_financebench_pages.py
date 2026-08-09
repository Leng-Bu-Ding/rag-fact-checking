from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.financebench import (
    read_document_manifest,
    read_questions_jsonl,
    write_document_manifest,
)

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.casefold())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate FinanceBench gold page indexes against parsed source PDFs."
    )
    parser.add_argument(
        "--questions", default="data/raw/financebench/questions.jsonl"
    )
    parser.add_argument(
        "--manifest", default="data/raw/financebench/document_manifest.json"
    )
    parser.add_argument(
        "--output", default="results/financebench_page_validation.json"
    )
    parser.add_argument(
        "--quarantine-invalid-alternates",
        action="store_true",
        help="Mark alternate PDFs with page mismatches as validation_failed.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    questions = read_questions_jsonl(project_path(args.questions))
    all_documents = read_document_manifest(project_path(args.manifest))
    documents = {
        item.doc_name: item
        for item in all_documents
        if item.status == "parsed"
    }
    evidence = [
        (question.financebench_id, item)
        for question in questions
        if question.doc_name in documents
        for item in question.evidence
    ]
    page_cache: dict[str, list[str]] = {}
    records = []
    for question_id, item in evidence:
        document = documents[item.doc_name]
        if item.doc_name not in page_cache:
            with pymupdf.open(PROJECT_ROOT / document.local_path) as pdf:
                page_cache[item.doc_name] = [normalize(page.get_text("text")) for page in pdf]
        pages = page_cache[item.doc_name]
        needle = normalize(item.evidence_text)
        found_pages = [index + 1 for index, text in enumerate(pages) if needle and needle in text]
        records.append(
            {
                "financebench_id": question_id,
                "doc_name": item.doc_name,
                "source_page_index": item.source_page_index,
                "expected_pdf_page": item.page_number,
                "exact_normalized_match_pages": found_pages,
                "expected_page_match": item.page_number in found_pages,
                "unconverted_page_match": item.source_page_index in found_pages,
            }
        )
    report = {
        "schema_version": 1,
        "method": "lowercase alphanumeric exact substring; gold text is validation-only",
        "evidence_count": len(records),
        "expected_page_matches": sum(item["expected_page_match"] for item in records),
        "unconverted_page_matches": sum(item["unconverted_page_match"] for item in records),
        "matched_elsewhere": sum(bool(item["exact_normalized_match_pages"]) for item in records),
        "unmatched": sum(not item["exact_normalized_match_pages"] for item in records),
        "records": records,
    }
    invalid_documents = sorted(
        {
            item["doc_name"]
            for item in records
            if not item["expected_page_match"]
        }
    )
    report["invalid_documents"] = invalid_documents
    if args.quarantine_invalid_alternates:
        invalid_alternates = {
            name
            for name in invalid_documents
            if documents[name].original_source_url is not None
        }
        updated = [
            replace(
                item,
                status="validation_failed",
                error="alternate PDF does not preserve FinanceBench gold page alignment",
            )
            if item.doc_name in invalid_alternates
            else item
            for item in all_documents
        ]
        write_document_manifest(updated, project_path(args.manifest))
        report["quarantined_alternate_documents"] = sorted(invalid_alternates)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.financebench import (
    FinanceBenchAdapter,
    apply_alternate_urls,
    build_finance_chunks,
    download_documents,
    read_document_manifest,
    read_questions_jsonl,
    write_document_manifest,
    write_questions_jsonl,
)
from src.data.jsonl import write_chunks_jsonl
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch FinanceBench metadata, download source PDFs, and build page chunks."
    )
    parser.add_argument(
        "command",
        choices=("fetch", "apply-alternates", "download", "build", "all"),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "financebench.yaml"),
    )
    parser.add_argument(
        "--only-alternates",
        action="store_true",
        help="Download only failed documents with a configured alternate URL.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def fetch(settings: dict) -> dict:
    cache_dir = settings["dataset"].get("cache_dir")
    adapter = FinanceBenchAdapter.from_huggingface(cache_dir=cache_dir)
    questions_path = project_path(settings["questions_path"])
    manifest_path = project_path(settings["document_manifest_path"])
    documents = adapter.documents()
    write_questions_jsonl(adapter.questions, questions_path)
    write_document_manifest(documents, manifest_path)
    return {
        "questions": len(adapter.questions),
        "documents": len(documents),
        "companies": len({item.company for item in adapter.questions}),
        "evidence_refs": sum(len(item.evidence) for item in adapter.questions),
        "question_types": dict(sorted(Counter(item.question_type for item in adapter.questions).items())),
        "questions_path": display_path(questions_path),
        "manifest_path": display_path(manifest_path),
    }


def download(settings: dict, *, only_alternates: bool = False) -> dict:
    manifest_path = project_path(settings["document_manifest_path"])
    documents = read_document_manifest(manifest_path)
    config = settings["download"]
    targets = (
        [item for item in documents if item.status == "failed" and item.original_source_url]
        if only_alternates
        else documents
    )
    downloaded = download_documents(
        targets,
        PROJECT_ROOT,
        workers=int(config["workers"]),
        timeout=int(config["timeout_seconds"]),
    )
    replacements = {item.doc_name: item for item in downloaded}
    updated = [replacements.get(item.doc_name, item) for item in documents]
    write_document_manifest(updated, manifest_path)
    statuses = Counter(item.status for item in updated)
    return {
        "documents": len(updated),
        "attempted_documents": len(targets),
        "statuses": dict(sorted(statuses.items())),
        "failures": [
            {"doc_name": item.doc_name, "url": item.source_url, "error": item.error}
            for item in updated
            if item.status == "failed"
        ],
        "manifest_path": display_path(manifest_path),
    }


def apply_alternates(settings: dict) -> dict:
    manifest_path = project_path(settings["document_manifest_path"])
    alternates_path = project_path(settings["alternate_urls_path"])
    documents = read_document_manifest(manifest_path)
    alternate_urls = json.loads(alternates_path.read_text(encoding="utf-8"))
    updated = apply_alternate_urls(documents, alternate_urls)
    write_document_manifest(updated, manifest_path)
    return {
        "configured_alternates": len(alternate_urls),
        "pending_alternates": sum(
            item.doc_name in alternate_urls and item.status == "failed" for item in updated
        ),
        "alternate_urls_path": display_path(alternates_path),
    }


def build(settings: dict) -> dict:
    manifest_path = project_path(settings["document_manifest_path"])
    chunks_path = project_path(settings["chunks_path"])
    documents = read_document_manifest(manifest_path)
    config = settings["chunking"]
    chunks, updated = build_finance_chunks(
        documents,
        PROJECT_ROOT,
        chunk_words=int(config["chunk_words"]),
        overlap_words=int(config["overlap_words"]),
    )
    write_chunks_jsonl(chunks, chunks_path)
    write_document_manifest(updated, manifest_path)
    statuses = Counter(item.status for item in updated)
    return {
        "documents": len(updated),
        "statuses": dict(sorted(statuses.items())),
        "pages": sum(
            item.page_count or 0 for item in updated if item.status == "parsed"
        ),
        "chunks": len(chunks),
        "chunks_with_questions": sum(bool(item.question) for item in chunks),
        "chunks_with_answers": sum(bool(item.answer) for item in chunks),
        "chunks_with_gold_labels": sum(item.contains_supporting_fact for item in chunks),
        "chunks_path": display_path(chunks_path),
    }


def main() -> None:
    args = parse_args()
    settings = load_config(args.config)["financebench"]
    report: dict[str, dict] = {}
    if args.command in {"fetch", "all"}:
        report["fetch"] = fetch(settings)
    if args.command in {"apply-alternates", "all"}:
        report["apply_alternates"] = apply_alternates(settings)
    if args.command in {"download", "all"}:
        report["download"] = download(settings, only_alternates=args.only_alternates)
    if args.command in {"build", "all"}:
        report["build"] = build(settings)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

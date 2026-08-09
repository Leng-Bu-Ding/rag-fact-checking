from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunk_io import read_chunks_jsonl
from src.data.financebench import read_document_manifest, read_questions_jsonl
from src.pipelines.dense import (
    build_dense_index,
    build_dense_index_reusing,
    sha256_file,
)
from src.pipelines.financebench import (
    build_finance_rankings,
    covered_questions,
    evaluate_finance_systems,
    fuse_finance_rankings,
    rerank_finance,
)
from src.pipelines.retrieval_experiment import write_report
from src.retrieval.dense import DenseIndex, SentenceTransformerEncoder, chunk_text
from src.retrieval.reranker import (
    CrossEncoderReranker,
    TransformersCrossEncoderScorer,
)
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinanceBench retrieval over real source PDFs."
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "financebench.yaml")
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def make_encoder(config: dict[str, Any]) -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder(
        str(config["model_name"]),
        revision=config.get("revision"),
        device=str(config["device"]),
        cache_folder=config.get("cache_folder"),
        local_files_only=bool(config.get("local_files_only", False)),
    )


def load_or_build_index(
    chunks: list,
    chunks_path: Path,
    index_dir: Path,
    encoder: SentenceTransformerEncoder,
    config: dict[str, Any],
    *,
    rebuild: bool,
) -> DenseIndex:
    chunks_hash = sha256_file(chunks_path)
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists() and not rebuild:
        index, manifest = DenseIndex.load(index_dir)
        if manifest.get("source", {}).get("chunks_sha256") == chunks_hash:
            return index
        index, reuse = build_dense_index_reusing(
            chunks,
            encoder,
            index,
            batch_size=int(config["batch_size"]),
            include_title=bool(config["include_title"]),
            normalize=bool(config["normalize"]),
        )
    else:
        index = build_dense_index(
            chunks,
            encoder,
            batch_size=int(config["batch_size"]),
            include_title=bool(config["include_title"]),
            normalize=bool(config["normalize"]),
        )
        reuse = {"reused_chunk_count": 0, "encoded_chunk_count": len(chunks)}
    index.save(
        index_dir,
        {
            "source": {
                "chunks_path": display_path(chunks_path),
                "chunks_sha256": chunks_hash,
            },
            "embedding": {
                **encoder.metadata,
                "batch_size": int(config["batch_size"]),
                "include_title": bool(config["include_title"]),
            },
            "incremental_build": reuse,
        },
    )
    return index


def main() -> None:
    args = parse_args()
    settings = load_config(args.config)["financebench"]
    questions_path = project_path(settings["questions_path"])
    manifest_path = project_path(settings["document_manifest_path"])
    chunks_path = project_path(settings["chunks_path"])
    index_dir = project_path(settings["dense_index_dir"])
    output_path = project_path(args.output or settings["retrieval_output_path"])

    all_questions = read_questions_jsonl(questions_path)
    documents = read_document_manifest(manifest_path)
    chunks = read_chunks_jsonl(chunks_path)
    questions = covered_questions(all_questions, chunks)
    if not questions:
        raise ValueError("no FinanceBench questions have parsed source documents")

    dense_config = settings["dense"]
    encoder = make_encoder(dense_config)
    index_started = time.perf_counter()
    dense_index = load_or_build_index(
        chunks,
        chunks_path,
        index_dir,
        encoder,
        dense_config,
        rebuild=args.rebuild_index,
    )
    index_seconds = time.perf_counter() - index_started
    query_started = time.perf_counter()
    query_matrix = encoder.encode_queries(
        [item.question for item in questions],
        batch_size=int(dense_config["batch_size"]),
    )
    query_seconds = time.perf_counter() - query_started
    query_vectors = {
        item.financebench_id: vector
        for item, vector in zip(questions, query_matrix, strict=True)
    }

    top_ks = [int(value) for value in settings["evaluation"]["top_ks"]]
    ranking_depth = int(settings["evaluation"]["ranking_depth"])
    bm25_config = {
        "include_title": bool(settings["bm25"]["include_title"]),
        "k1": float(settings["bm25"]["k1"]),
        "b": float(settings["bm25"]["b"]),
        "epsilon": float(settings["bm25"]["epsilon"]),
    }
    scope_reports: dict[str, Any] = {}
    for scope in ("global", "document"):
        base, timings = build_finance_rankings(
            chunks,
            questions,
            dense_index,
            query_vectors,
            corpus_scope=scope,
            ranking_depth=ranking_depth,
            bm25_config=bm25_config,
        )
        hybrid_config = settings["hybrid"]
        fusion_started = time.perf_counter()
        hybrid = fuse_finance_rankings(
            base,
            questions,
            candidate_k=int(hybrid_config["candidate_k"]),
            rrf_k=int(hybrid_config["rrf_k"]),
            top_k=ranking_depth,
        )
        fusion_seconds = time.perf_counter() - fusion_started
        rankings = {**base, "hybrid": hybrid}
        reranker_record = None
        reranker_config = settings["reranker"]
        if bool(reranker_config["enabled"]) and not args.skip_reranker:
            candidate_k = int(reranker_config["candidate_k"])
            candidates = {
                key: value[:candidate_k] for key, value in hybrid.items()
            }
            scorer = TransformersCrossEncoderScorer(
                str(reranker_config["model_name"]),
                cache_dir=reranker_config.get("cache_dir"),
                device=str(reranker_config["device"]),
                local_files_only=bool(reranker_config["local_files_only"]),
                max_length=int(reranker_config["max_length"]),
            )
            reranker = CrossEncoderReranker(
                scorer, batch_size=int(reranker_config["batch_size"])
            )
            rankings["reranker"], reranker_timing = rerank_finance(
                reranker, questions, candidates, top_k=max(top_ks)
            )
            reranker_record = {
                "metadata": reranker.metadata,
                "candidate_k": candidate_k,
                "timing": reranker_timing,
            }
        scope_reports[scope] = {
            "timings": {
                **timings,
                "hybrid_fusion_total_seconds": round(fusion_seconds, 6),
                "hybrid_fusion_milliseconds_per_query": round(
                    fusion_seconds * 1000 / len(questions), 6
                ),
            },
            "reranker": reranker_record,
            "systems": evaluate_finance_systems(
                questions, rankings, top_ks=top_ks
            ),
        }

    parsed_documents = [item for item in documents if item.status == "parsed"]
    report = {
        "schema_version": 1,
        "experiment": "financebench_real_pdf_retrieval",
        "source": {
            "dataset": "PatronusAI/financebench",
            "questions_path": display_path(questions_path),
            "document_manifest_path": display_path(manifest_path),
            "chunks_path": display_path(chunks_path),
            "chunks_sha256": sha256_file(chunks_path),
            "dense_index_manifest": display_path(index_dir / "manifest.json"),
        },
        "coverage": {
            "public_question_count": len(all_questions),
            "evaluated_question_count": len(questions),
            "question_coverage": round(len(questions) / len(all_questions), 8),
            "public_document_count": len(documents),
            "parsed_document_count": len(parsed_documents),
            "document_coverage": round(len(parsed_documents) / len(documents), 8),
            "parsed_page_count": sum(item.page_count or 0 for item in parsed_documents),
            "chunk_count": len(chunks),
            "failed_documents": [
                {"doc_name": item.doc_name, "error": item.error}
                for item in documents
                if item.status not in {"downloaded", "parsed"}
            ],
        },
        "leakage_checks": {
            "chunks_with_question": sum(bool(item.question) for item in chunks),
            "chunks_with_answer": sum(bool(item.answer) for item in chunks),
            "chunks_with_gold_label": sum(item.contains_supporting_fact for item in chunks),
        },
        "config": settings,
        "shared_timings": {
            "index_load_or_build_seconds": round(index_seconds, 6),
            "query_encoding_total_seconds": round(query_seconds, 6),
            "query_encoding_milliseconds_per_query": round(
                query_seconds * 1000 / len(questions), 6
            ),
        },
        "scopes": scope_reports,
    }
    write_report(report, output_path)
    print(
        json.dumps(
            {
                "output": display_path(output_path),
                "coverage": report["coverage"],
                "metrics": {
                    scope: {
                        system: evaluation["metrics"]
                        for system, evaluation in values["systems"].items()
                    }
                    for scope, values in scope_reports.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunk_io import read_chunks_jsonl
from src.evaluation.retrieval import covered_gold_facts, gold_facts_for_sample
from src.pipelines.bm25 import CorpusScope, group_chunks_by_sample, question_for_sample
from src.pipelines.dense import (
    build_dense_index,
    compare_reports,
    evaluate_dense,
    search_dense,
    sha256_file,
    write_json_report,
)
from src.retrieval.dense import DenseIndex, SentenceTransformerEncoder
from src.utils.config import load_config


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "dense.yaml"),
    )
    parser.add_argument("--index-dir", default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, evaluate, or inspect dense FAISS retrieval."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build and persist index.")
    _add_common_options(build_parser)
    build_parser.add_argument("--chunks", default=None)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate supporting-fact retrieval."
    )
    _add_common_options(evaluate_parser)
    evaluate_parser.add_argument(
        "--corpus-scope", choices=("global", "sample"), default=None
    )
    evaluate_parser.add_argument("--top-ks", type=int, nargs="+", default=None)
    evaluate_parser.add_argument("--output", default=None)
    evaluate_parser.add_argument("--bm25-report", default=None)
    evaluate_parser.add_argument("--comparison-output", default=None)

    query_parser = subparsers.add_parser("query", help="Inspect ranked chunks.")
    _add_common_options(query_parser)
    query_parser.add_argument("--question", default=None)
    query_parser.add_argument("--sample-id", default=None)
    query_parser.add_argument(
        "--corpus-scope", choices=("global", "sample"), default=None
    )
    query_parser.add_argument("--top-k", type=int, default=None)
    query_parser.add_argument("--show-gold", action="store_true")
    return parser.parse_args()


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_settings(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config = load_config(args.config)["dense"]
    index_dir = _resolve_project_path(args.index_dir or config["index_dir"])
    return config, index_dir


def _encoder(config: dict[str, Any]) -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder(
        str(config["model_name"]),
        revision=config.get("revision"),
        device=str(config["device"]),
        seed=int(config["seed"]),
        cache_folder=config.get("cache_folder"),
        local_files_only=bool(config.get("local_files_only", False)),
    )


def run_build(args: argparse.Namespace) -> None:
    config, index_dir = _load_settings(args)
    chunks_path = _resolve_project_path(args.chunks or config["chunks_path"])
    chunks = read_chunks_jsonl(chunks_path)
    encoder = _encoder(config)
    index = build_dense_index(
        chunks,
        encoder,
        batch_size=int(config["batch_size"]),
        include_title=bool(config["include_title"]),
        normalize=bool(config["normalize"]),
    )
    manifest = {
        "source": {
            "chunks_path": _display_path(chunks_path),
            "chunks_sha256": sha256_file(chunks_path),
        },
        "embedding": {
            **encoder.metadata,
            "batch_size": int(config["batch_size"]),
            "include_title": bool(config["include_title"]),
        },
    }
    index.save(index_dir, manifest)
    print(
        json.dumps(
            {
                "chunk_count": index.chunk_count,
                "dimension": index.dimension,
                "normalize": index.normalize,
                "index_dir": _display_path(index_dir),
                "encoder": encoder.metadata,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_evaluate(args: argparse.Namespace) -> None:
    config, index_dir = _load_settings(args)
    index, manifest = DenseIndex.load(index_dir)
    encoder = _encoder(config)
    if encoder.metadata["dimension"] != index.dimension:
        raise ValueError("configured encoder dimension does not match index")
    corpus_scope: CorpusScope = args.corpus_scope or config["corpus_scope"]
    report = evaluate_dense(
        index,
        encoder,
        corpus_scope=corpus_scope,
        top_ks=args.top_ks or config["top_ks"],
        batch_size=int(config["batch_size"]),
    )
    report["source"] = manifest["source"]
    report["index_manifest"] = _display_path(index_dir / "manifest.json")
    output_path = _resolve_project_path(args.output or config["results_path"])
    write_json_report(report, output_path)

    bm25_path_value = args.bm25_report or config.get("bm25_report_path")
    comparison_path_value = (
        args.comparison_output or config.get("comparison_path")
    )
    comparison_path = None
    if bm25_path_value and comparison_path_value:
        bm25_path = _resolve_project_path(bm25_path_value)
        bm25_report = json.loads(bm25_path.read_text(encoding="utf-8"))
        comparison = compare_reports(bm25_report, report)
        comparison_path = _resolve_project_path(comparison_path_value)
        write_json_report(comparison, comparison_path)

    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "metrics": report["metrics"],
                "corpus_scope": corpus_scope,
                "output": _display_path(output_path),
                "comparison": (
                    _display_path(comparison_path) if comparison_path else None
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_query(args: argparse.Namespace) -> None:
    config, index_dir = _load_settings(args)
    index, _ = DenseIndex.load(index_dir)
    encoder = _encoder(config)
    grouped = group_chunks_by_sample(index.chunks)
    corpus_scope: CorpusScope = args.corpus_scope or config["corpus_scope"]
    top_k = args.top_k if args.top_k is not None else max(config["top_ks"])

    if args.show_gold and not args.sample_id:
        raise ValueError("--show-gold requires --sample-id")
    if corpus_scope == "sample" and not args.sample_id:
        raise ValueError("sample corpus scope requires --sample-id")
    if args.sample_id and args.sample_id not in grouped:
        raise ValueError(f"unknown sample_id: {args.sample_id}")

    question = args.question
    if question is None and args.sample_id:
        question = question_for_sample(index.chunks, args.sample_id)
    if not question:
        raise ValueError("provide --question or --sample-id")

    search_index = (
        index.subset(grouped[args.sample_id])
        if corpus_scope == "sample" and args.sample_id
        else index
    )
    results = search_dense(
        search_index,
        encoder,
        question,
        top_k=top_k,
        batch_size=int(config["batch_size"]),
    )
    gold_facts = (
        gold_facts_for_sample(index.chunks, args.sample_id)
        if args.show_gold and args.sample_id
        else frozenset()
    )

    print(f"Question: {question}")
    print(f"Corpus scope: {corpus_scope} ({search_index.chunk_count} chunks)")
    print("-" * 100)
    for result in results:
        covered = covered_gold_facts(result.chunk, gold_facts)
        gold_label = " gold=yes" if covered else (" gold=no" if args.show_gold else "")
        print(
            f"[{result.rank}] score={result.score:.6f}{gold_label} "
            f"sample={result.chunk.sample_id} title={result.chunk.title}"
        )
        print(
            f"    chunk={result.chunk.chunk_id} "
            f"sentences={result.chunk.sentence_ids}"
        )
        print(f"    {result.chunk.text}")
        if covered:
            labels = ", ".join(
                f"{title}[{sentence_id}]"
                for _, title, sentence_id in sorted(covered)
            )
            print(f"    covered_gold={labels}")
        print("-" * 100)


def main() -> None:
    args = parse_args()
    if args.command == "build":
        run_build(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_query(args)


if __name__ == "__main__":
    main()

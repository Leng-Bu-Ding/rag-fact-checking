from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunk_io import read_chunks_jsonl
from src.evaluation.retrieval import covered_gold_facts, gold_facts_for_sample
from src.pipelines.bm25 import (
    CorpusScope,
    evaluate_bm25,
    group_chunks_by_sample,
    question_for_sample,
    search_bm25,
    write_evaluation_report,
)
from src.utils.config import load_config


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "bm25.yaml"),
    )
    parser.add_argument("--chunks", default=None)
    parser.add_argument("--corpus-scope", choices=("global", "sample"), default=None)
    parser.add_argument("--top-k", type=int, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate or inspect deterministic BM25 chunk retrieval."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate HotpotQA supporting-fact retrieval.",
    )
    _add_common_options(evaluate_parser)
    evaluate_parser.add_argument("--output", default=None)
    evaluate_parser.add_argument("--top-ks", type=int, nargs="+", default=None)

    query_parser = subparsers.add_parser(
        "query",
        help="Inspect ranked chunks for one question.",
    )
    _add_common_options(query_parser)
    query_parser.add_argument("--question", default=None)
    query_parser.add_argument("--sample-id", default=None)
    query_parser.add_argument(
        "--show-gold",
        action="store_true",
        help="Show evaluation-only gold labels; requires --sample-id.",
    )
    return parser.parse_args()


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_settings(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config = load_config(args.config)["bm25"]
    chunks_path = _resolve_project_path(args.chunks or config["chunks_path"])
    return config, chunks_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_evaluate(args: argparse.Namespace) -> None:
    config, chunks_path = _load_settings(args)
    chunks = read_chunks_jsonl(chunks_path)
    corpus_scope: CorpusScope = args.corpus_scope or config["corpus_scope"]
    top_ks = args.top_ks or config["top_ks"]
    report = evaluate_bm25(
        chunks,
        corpus_scope=corpus_scope,
        top_ks=top_ks,
        include_title=bool(config["include_title"]),
        k1=float(config["k1"]),
        b=float(config["b"]),
        epsilon=float(config["epsilon"]),
    )
    report["source"] = {
        "chunks_path": _display_path(chunks_path),
        "chunks_sha256": _sha256(chunks_path),
    }
    output_path = _resolve_project_path(args.output or config["results_path"])
    write_evaluation_report(report, output_path)
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "metrics": report["metrics"],
                "corpus_scope": corpus_scope,
                "output": _display_path(output_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_query(args: argparse.Namespace) -> None:
    config, chunks_path = _load_settings(args)
    chunks = read_chunks_jsonl(chunks_path)
    grouped = group_chunks_by_sample(chunks)
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
        question = question_for_sample(chunks, args.sample_id)
    if not question:
        raise ValueError("provide --question or --sample-id")

    corpus = grouped[args.sample_id] if corpus_scope == "sample" else chunks
    results = search_bm25(
        corpus,
        question,
        top_k=top_k,
        include_title=bool(config["include_title"]),
        k1=float(config["k1"]),
        b=float(config["b"]),
        epsilon=float(config["epsilon"]),
    )
    gold_facts = (
        gold_facts_for_sample(chunks, args.sample_id)
        if args.show_gold and args.sample_id
        else frozenset()
    )

    print(f"Question: {question}")
    print(f"Corpus scope: {corpus_scope} ({len(corpus)} chunks)")
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
    if args.command == "evaluate":
        run_evaluate(args)
    else:
        run_query(args)


if __name__ == "__main__":
    main()

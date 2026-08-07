from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunking import chunk_samples
from src.data.jsonl import write_chunks_jsonl
from src.data.load_hotpotqa import load_hotpotqa_samples
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build metadata-preserving HotpotQA chunks."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    sample_size = args.sample_size or data_config["sample_size"]

    samples = load_hotpotqa_samples(
        sample_size=sample_size,
        subset=data_config["primary_subset"],
        split=data_config["split"],
        cache_dir=str(PROJECT_ROOT / data_config["cache_dir"]),
    )
    chunks = chunk_samples(
        samples,
        chunk_size=preprocessing_config["chunk_size"],
        chunk_overlap=preprocessing_config["chunk_overlap"],
        min_text_length=preprocessing_config["min_text_length"],
    )
    output_path = Path(args.output).resolve()
    write_chunks_jsonl(chunks, output_path)

    document_count = sum(len(sample.documents) for sample in samples)
    supporting_chunk_count = sum(
        chunk.contains_supporting_fact for chunk in chunks
    )
    average_chunk_length = (
        sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0.0
    )
    stats = {
        "samples": len(samples),
        "documents": document_count,
        "chunks": len(chunks),
        "supporting_chunks": supporting_chunk_count,
        "average_chunk_length": round(average_chunk_length, 2),
        "output": str(output_path),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

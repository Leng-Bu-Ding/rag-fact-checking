from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_hotpotqa import load_hotpotqa_samples
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and print HotpotQA samples.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    sample_size = args.sample_size or config["data"]["sample_size"]
    subset = config["data"]["primary_subset"]
    split = config["data"]["split"]
    cache_dir = str(PROJECT_ROOT / config["data"]["cache_dir"])

    samples = load_hotpotqa_samples(
        sample_size=sample_size,
        subset=subset,
        split=split,
        cache_dir=cache_dir,
    )

    print(f"Loaded {len(samples)} HotpotQA samples from split={split}, subset={subset}.")
    print("-" * 80)

    for idx, sample in enumerate(samples, start=1):
        print(f"[Sample {idx}] id={sample.sample_id}")
        print(f"Question: {sample.question}")
        print(f"Answer: {sample.answer}")
        print(f"Supporting facts: {sample.supporting_facts[:3]}")
        print("Context preview:")

        for doc in sample.documents[:2]:
            preview = doc["text"][:240].replace("\n", " ")
            print(f"  - {doc['title']}: {preview}")

        print("-" * 80)


if __name__ == "__main__":
    main()

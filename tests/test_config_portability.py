from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("config_path", "section", "cache_key"),
    [
        ("configs/dense.yaml", ("dense",), "cache_folder"),
        (
            "configs/hotpotqa_retrieval.yaml",
            ("hotpotqa_retrieval", "dense"),
            "cache_folder",
        ),
        (
            "configs/hotpotqa_retrieval.yaml",
            ("hotpotqa_retrieval", "reranker"),
            "cache_dir",
        ),
    ],
)
def test_public_model_configs_do_not_require_a_machine_specific_cache(
    config_path: str,
    section: tuple[str, ...],
    cache_key: str,
) -> None:
    config = load_config(PROJECT_ROOT / config_path)
    for key in section:
        config = config[key]

    assert config[cache_key] is None
    assert config["local_files_only"] is False

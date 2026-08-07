from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.data.chunking import DocumentChunk


def write_chunks_jsonl(
    chunks: Iterable[DocumentChunk],
    output_path: str | Path,
) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(
                json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    return count

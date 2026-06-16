from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.models import ParsedChunk
from ingestion.parser import parse
from scripts.chat_cli import C


def main() -> int:
    args = _parse_args()

    path = Path(args.filepath)

    if not path.exists():
        C.red(f"[error] file does not exist: {path}")
        return 1

    try:
        filename, content = _load_file(path)
        chunks = parse(filename, content)

        if not chunks:
            C.yellow("[warn] no chunks generated")
            return 0

        report = _analyze_chunks(chunks)

        _render(chunks, report, args)

    except Exception as exc:
        C.red(f"[error] failed to process file: {exc}")
        return 1

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ingestion chunking behavior"
    )

    parser.add_argument(
        "filepath",
        help="Path to file to inspect",
    )

    group = parser.add_argument_group()

    group.add_argument(
        "--show-overlap",
        action="store_true",
        help="Show overlap between adjacent chunks",
    )

    group.add_argument(
        "--raw",
        action="store_true",
        help="Print only chunk content (no metadata)",
    )

    group.add_argument(
        "--json",
        action="store_true",
        help="Output chunks as JSON",
    )

    return parser.parse_args()


def _load_file(path: Path) -> tuple[str, bytes]:
    """
    Loads a file and returns:
    - filename (str)
    - raw content (bytes)
    """
    filename: str = path.name
    content: bytes = path.read_bytes()

    return filename, content


@dataclass
class ChunkReport:
    total: int
    avg_size: float
    min_size: int
    max_size: int


def _analyze_chunks(chunks: list[ParsedChunk]) -> ChunkReport:
    """
    Compute summary statistics for chunks.
    """

    if not chunks:
        return ChunkReport(
            total=0,
            avg_size=0.0,
            min_size=0,
            max_size=0,
        )

    total_chunks: int = len(chunks)
    min_size, max_size, total_size = len(chunks[0].content),0, 0

    for chunk in chunks:
        chunk_size = len(chunk.content)

        if chunk_size < min_size:
            min_size = chunk_size

        if chunk_size > max_size:
            max_size = chunk_size

        total_size += chunk_size
    
    return ChunkReport(
        total=total_chunks,
        avg_size=total_size/total_chunks,
        min_size=min_size,
        max_size=max_size
    )



def _render(
    chunks: list[ParsedChunk],
    report: ChunkReport,
    args: argparse.Namespace,
) -> None:
    """
    Handles all output modes:
    - pretty
    - raw
    - json
    """
    raise NotImplementedError

if __name__ == "__main__":
    raise SystemExit(main())
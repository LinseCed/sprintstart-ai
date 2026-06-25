from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cli_colors import C
from rich.align import Align
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from ingestion.models import ParsedChunk  # noqa: E402
from ingestion.parser import parse  # noqa: E402

console = Console()


def main() -> int:

    args = _parse_args()

    path = Path(args.filepath)

    if not path.exists():
        print(C.red(f"[error] file does not exist: {path}"))
        return 1

    try:
        filename, content = _load_file(path)
        chunks = parse(filename, content)

        if not chunks:
            print(C.yellow("[warn] no chunks generated"))
            return 0

        report = _analyze_chunks(chunks)

        _render(chunks, report, args)

    except Exception as exc:
        print(C.red(f"[error] failed to process file: {exc}"))
        return 1

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ingestion chunking behavior")

    parser.add_argument(
        "filepath",
        help="Path to file to inspect",
    )

    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
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
    min_size, max_size, total_size = len(chunks[0].content), 0, 0

    for chunk in chunks:
        chunk_size = len(chunk.content)

        if chunk_size < min_size:
            min_size = chunk_size

        if chunk_size > max_size:
            max_size = chunk_size

        total_size += chunk_size

    return ChunkReport(
        total=total_chunks,
        avg_size=total_size / total_chunks,
        min_size=min_size,
        max_size=max_size,
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
    if args.raw:
        console.rule(title="Raw Chunks", characters="=")
        print_chunks(chunks, pretty=False, show_overlap=args.show_overlap)

    elif args.json:
        data = [chunk_to_dict(chunk) for chunk in chunks]
        print(json.dumps(data, indent=2, ensure_ascii=False))

    else:
        console.rule(title="Detailed Chunks", characters="=")
        print_chunks(chunks, pretty=True, show_overlap=args.show_overlap)

    console.rule(title="Overall Report", characters="=")
    report_table = Table(expand=False)
    report_table.add_column("Total Chunks", justify="center")
    report_table.add_column("Avg. Chunk Size", justify="center")
    report_table.add_column("Min Chunk Size", justify="center")
    report_table.add_column("Max Chunk Size", justify="center")
    report_table.add_row(
        f" {report.total}",
        f" {report.avg_size}",
        f" {report.min_size}",
        f" {report.max_size}",
    )

    console.print(Align.center(report_table))
    print("")


def print_chunks(
    chunks: list[ParsedChunk], pretty: bool = True, show_overlap: bool = False
) -> None:
    """Print all generated chunks to the terminal.

    Depending on the selected mode, the function prints chunk metadata,
    content, and optionally highlights overlap regions shared with
    adjacent chunks.

    Args:
        chunks (list[ParsedChunk]):
            Chunks produced by the ingestion pipeline.

        pretty (bool, optional):
            Whether to render chunk metadata tables in addition to the
            chunk content. Defaults to True.

        show_overlap (bool, optional):
            Whether overlap regions should be highlighted in the chunk
            content output. Defaults to False.
    """
    
    for index, chunk in enumerate(chunks):
        print("")
        isPdf: bool = (chunk.kind == "pdf")
        isCode: bool = (chunk.kind =="code")
        if isPdf:
            console.rule(title=f"Chunk {chunk.metadata['chunk_index']} - Page {chunk.metadata['page_number']}", characters="=")
        elif isCode:
            console.rule(title=f"Chunk {chunk.metadata['chunk_index']} - {chunk.metadata['symbol_kind']}: {chunk.metadata['symbol_name']}", characters="=")
        else:
            console.rule(title=f"Chunk {chunk.metadata['chunk_index']}", characters="=")
        # compute overlap
        if index > 0:
            start_overlap: int = get_overlap_count(
                chunks[index - 1].content, chunks[index].content
            )
        else:
            start_overlap = 0
        try:
            end_overlap: int = get_overlap_count(
                chunks[index].content, chunks[index + 1].content
            )
        except IndexError:
            end_overlap = 0

        if pretty:
            metadata_table = Table(expand=False)
            metadata_table.add_column("chunk kind", justify="center")
            metadata_table.add_column("filename", justify="center")
            metadata_table.add_column("type", justify="center")
            if isPdf:
                metadata_table.add_column("global-chunk-index", justify="center")
            elif isCode:
                metadata_table.add_column("symbol-name",justify="center")
            metadata_table.add_column("character count", justify="center")
            metadata_table.add_column("front-overlap", justify="center")
            metadata_table.add_column("end-overlap", justify="center")
            row = [
                chunk.kind,
                chunk.metadata["filename"],
                chunk.metadata["type"],
            ]

            if isPdf:
                row.append(str(chunk.metadata["global_pdf_chunk_index"]))

            if isCode:
                row.append(chunk.metadata["symbol_name"])

            row.extend(
                [
                    str(len(chunk.content)),
                    str(start_overlap),
                    str(end_overlap),
                ]
            )

            metadata_table.add_row(*row)
            console.print(Align.center(metadata_table))
        print("")
        print(C.bold("content:"))
        if show_overlap:
            print(C.green(chunk.content[:start_overlap]), end="")
            print(
                C.dim(chunk.content[start_overlap : len(chunk.content) - end_overlap]),
                end="",
            )
            print(C.green(chunk.content[len(chunk.content) - end_overlap :]), end="")

        else:
            print(C.dim(chunk.content))

        print("")


def chunk_to_dict(chunk: ParsedChunk) -> dict[str, Any]:
    result = {}
    result["content"] = chunk.content
    result["kind"] = chunk.kind
    result["metadata"] = {}
    result["metadata"]["filename"] = chunk.metadata["filename"]
    result["metadata"]["type"] = chunk.metadata["type"]
    result["metadata"]["chunk_index"] = chunk.metadata["chunk_index"]
    result["metadata"]["total_chunks"] = chunk.metadata["total_chunks"]
    # TODO: add the specific metadata for pdf and code 

    return result  # type: ignore


def get_overlap_count(first_string: str, second_string: str) -> int:
    """Compute the overlap length between two chunk contents.

    The function searches for the largest suffix of the first string
    that matches a prefix of the second string. This corresponds to the
    overlap produced by the chunking logic when context is carried
    forward between adjacent chunks.

    Args:
        firstString (str):
            Content of the preceding chunk.

        secondString (str):
            Content of the following chunk.

    Returns:
        int:
            Length of the largest matching overlap in characters.
            Returns 0 if no overlap exists.
    """
    max_len = min(len(first_string), len(second_string))

    for size in range(max_len, 0, -1):
        if first_string[-size:] == second_string[:size]:
            return len(first_string[-size:])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

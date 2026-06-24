import re
from pathlib import Path

from ingestion.chunker import chunk_code
from ingestion.models import ParsedChunk
from ingestion.text_parser import parse_text
from ingestion.tree_sitter_parser import parse_with_tree_sitter

PATTERNS: dict[str, re.Pattern[str]] = {
    ".py": re.compile(r"^(async\s+def |def |class )"),
    ".js": re.compile(r"^(export\s+)?(async\s+)?(function|class|const|let|var)\b"),
    ".ts": re.compile(r"^(export\s+)?(async\s+)?(function|class|const|let|var)\b"),
    ".go": re.compile(r"func\s+\w+|type\s+\w+|const\s+\w+|var\s+\w+"),
}

def parse_code(filename: str, content: bytes) -> list[ParsedChunk]:

    try:
        symbols, preamble = parse_with_tree_sitter(filename, content)
    except Exception:
        return fallback_regex_parser(filename, content)

    if not symbols:
        return parse_text(filename, content)

    chunks: list[ParsedChunk] = []

    for symbol in symbols:
        full_content = (
            f"{preamble}\n\n{symbol.content}" if preamble else symbol.content
        )

        code_chunks = chunk_code(filename, full_content)

        for chunk in code_chunks:
            chunk.metadata = {
                **chunk.metadata,
                "symbol_name": symbol.name or "",
                "symbol_kind": symbol.kind,
            }

        chunks.extend(code_chunks)

    return chunks


def fallback_regex_parser(filename: str, content: bytes) -> list[ParsedChunk]:
    """Parse source code files into code-aware chunks.

    The parser detects top-level definitions such as functions and
    classes by matching with a regex pattern and creates self-contained chunks by prepending the file
    preamble (e.g. imports or module headers).

    If the language is unsupported or no top-level definitions are
    found, the parser falls back to the plain text parser.

    Args:
        filename (str):
            Name of the source file including extension.

        content (bytes):
            Raw file content as bytes.

    Returns:
        list[ParsedChunk]:
            Parsed code chunks with kind="code" or text chunks when
            fallback parsing is used.
    """

    file_suffix = Path(filename).suffix
    # Falls back to the text chunker if language is not supported
    if file_suffix not in PATTERNS.keys():
        return parse_text(filename, content)

    pattern: re.Pattern[str] = PATTERNS[file_suffix]
    lines = content.decode(encoding="utf-8", errors="replace").splitlines()

    boundaries: list[int] = compute_boundaries(lines, pattern)

    if not boundaries:
        # Falls back to the text chunker if no top level content could be found
        return parse_text(filename, content)

    preamble_end: int = boundaries[0]
    preamble: str = "\n".join(lines[:preamble_end]).strip()

    parsed_chunks: list[ParsedChunk] = []
    for start_line, end_line in zip(
        boundaries, boundaries[1:] + [len(lines)], strict=False
    ):
        content_body: str = "\n".join(lines[start_line:end_line]).strip()

        if not content_body:
            continue

        chunk_content: str = (
            f"{preamble}\n\n{content_body}" if preamble else content_body
        )
        for chunk in chunk_code(filename, chunk_content):
            parsed_chunks.append(chunk)

    return parsed_chunks


def compute_boundaries(lines: list[str], pattern: re.Pattern[str]) -> list[int]:
    """Compute top-level definition boundaries in source code.

    The function scans source code line-by-line and returns the
    indices of lines that start new top-level definitions.

    Only definitions without indentation are considered to avoid
    matching nested functions or classes.

    Args:
        lines (list[str]):
            Source code split into lines.

        pattern (re.Pattern[str]):
            Language-specific regex used to detect definitions.

    Returns:
        list[int]:
            Line indices representing the start of top-level
            definitions.
    """
    boundaries: list[int] = []
    for line_count, line in enumerate(lines):
        # filter for top level content without indentation
        if pattern.match(line) and line == line.lstrip():
            boundaries.append(line_count)
    return boundaries

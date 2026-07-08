import os
from pathlib import Path

from ingestion.models import ChunkKind, ParsedChunk
from ingestion.utils import build_metadata

try:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
except ValueError as err:
    raise ValueError("CHUNK_SIZE and CHUNK_OVERLAP must be an integer") from err

if chunk_overlap >= chunk_size:
    raise ValueError("chunk_overlap must be smaller than chunk_size")


def to_parsed_chunk(
    chunk_content: str,
    kind: ChunkKind,
    filename: str,
    chunk_index: int,
    total_chunks_amount: int,
    start_line: int | None = None,
):
    """Create a ParsedChunk with standard metadata.

    Args:
        chunk_content (str):
            Content of the chunk.

        kind (ChunkKind):
            Chunk type ("text", "code", "pdf", or "image").

        filename (str):
            Name of the source file.

        chunk_index (int):
            Zero-based index of the chunk within the source file.

        total_chunks_amount (int):
            Total number of chunks produced for the source file.

        start_line (int, optional):
            1-based line the chunk starts on in the source file, if known.

    Returns:
        ParsedChunk:
            Chunk instance with content, type and metadata.
    """
    metadata = {
        **build_metadata(Path(filename)),
        "chunk_index": str(chunk_index),
        "total_chunks": str(total_chunks_amount),
    }
    if start_line is not None:
        metadata["start_line"] = str(start_line)

    return ParsedChunk(
        content=chunk_content,
        kind=kind,
        metadata=metadata,
    )


def _paragraphs_with_start_lines(text: str) -> list[tuple[str, int]]:
    """Split text into non-empty paragraphs, keeping each one's start line.

    Mirrors ``text.split("\\n\\n")`` semantics (paragraphs separated by a
    blank line, stripped, empty ones dropped) while additionally tracking
    the 1-based line each surviving paragraph starts on in ``text``.
    """
    paragraphs: list[tuple[str, int]] = []
    offset = 0
    for part in text.split("\n\n"):
        stripped = part.strip()
        if stripped:
            # ``split("\n\n")`` only breaks on exactly two newlines, so runs of
            # 3+ newlines (extra blank lines) leave a leading "\n" attached to
            # the *next* segment (e.g. "A\n\n\nB".split("\n\n") ==
            # ["A", "\nB"]). Without accounting for it, the start offset would
            # be off by however many extra blank lines preceded the paragraph.
            leading_whitespace = len(part) - len(part.lstrip())
            start_offset = offset + leading_whitespace
            # ``count("\n", 0, start_offset)`` is the number of newlines before
            # the paragraph, i.e. its 0-based line index; +1 makes it 1-based.
            line_no = text.count("\n", 0, start_offset) + 1
            paragraphs.append((stripped, line_no))
        offset += len(part) + 2  # +2 accounts for the removed "\n\n" delimiter
    return paragraphs


def chunk_text(
    filename: str,
    text: str,
    chunk_size: int = chunk_size,
    chunk_overlap: int = chunk_overlap,
) -> list[ParsedChunk]:
    """Split text into paragraph-aware chunks.

    The function preserves paragraph boundaries by splitting on
    double newlines (``\\n\\n``). Paragraphs are accumulated until
    adding another paragraph would exceed the configured chunk size.

    When a chunk boundary is reached, the last paragraph is carried
    into the next chunk as overlap context. Paragraphs that exceed
    ``chunk_size`` on their own are split into overlapping character
    chunks by the given chunk_overlap.

    Args:
        filename (str):
            Name of the source file.

        text (str):
            Text content to split.

        chunk_size (int, optional):
            Maximum chunk size in characters.
            Defaults to the value configured via ``CHUNK_SIZE``.

        chunk_overlap (int, optional):
            Overlap used when hard-splitting oversized paragraphs.
            Defaults to the value configured via ``CHUNK_OVERLAP``.

    Returns:
        list[ParsedChunk]:
            Paragraph-aware text chunks with metadata.
    """

    raw_chunks: list[tuple[str, int]] = []

    paragraphs: list[tuple[str, int]] = _paragraphs_with_start_lines(text)
    current_chunk_content: list[tuple[str, int]] = []

    for paragraph, line in paragraphs:
        paragraph_length: int = len(paragraph)

        # hard split by character when paragraph itself exceeds chunk_size
        if paragraph_length > chunk_size:
            if current_chunk_content:
                chunk_content = "\n\n".join(p for p, _ in current_chunk_content)
                chunk_start_line = current_chunk_content[0][1]
                raw_chunks.append((chunk_content, chunk_start_line))
                overlap_text, overlap_line = current_chunk_content[-1]
                current_chunk_content = (
                    [(overlap_text, overlap_line)]
                    if len(overlap_text) < chunk_overlap
                    else []
                )

            combo: list[tuple[str, int]] = current_chunk_content + [(paragraph, line)]
            paragraph_with_overlap: str = "\n\n".join(p for p, _ in combo)
            combo_start_line: int = combo[0][1]

            start: int = 0
            while start < len(paragraph_with_overlap):
                newlines_before = paragraph_with_overlap.count("\n", 0, start)
                piece_start_line = combo_start_line + newlines_before
                raw_chunks.append(
                    (
                        paragraph_with_overlap[start : start + chunk_size],
                        piece_start_line,
                    )
                )
                start += chunk_size - chunk_overlap

            current_chunk_content = []
            continue

        # Would adding the paragraph exceed chunk_size?
        candidate_length: int = len(
            "\n\n".join([p for p, _ in current_chunk_content] + [paragraph])
        )

        # handle when current_chunk_content + paragraph exceeds chunk size
        if current_chunk_content and (candidate_length > chunk_size):
            chunk_content = "\n\n".join(p for p, _ in current_chunk_content)
            chunk_start_line = current_chunk_content[0][1]
            raw_chunks.append((chunk_content, chunk_start_line))
            overlap_text, overlap_line = current_chunk_content[-1]
            current_chunk_content = [(overlap_text, overlap_line)]

        # append whole paragraph to current_chunk_content
        current_chunk_content.append((paragraph, line))

    # if there is some content left, append it to the raw_chunks
    if current_chunk_content:
        chunk_content = "\n\n".join(p for p, _ in current_chunk_content)
        chunk_start_line = current_chunk_content[0][1]
        raw_chunks.append((chunk_content, chunk_start_line))

    total_chunks_amount = len(raw_chunks)

    return [
        to_parsed_chunk(
            chunk_content,
            "text",
            filename,
            chunk_index,
            total_chunks_amount,
            start_line=start_line,
        )
        for chunk_index, (chunk_content, start_line) in enumerate(raw_chunks)
    ]


def chunk_code(
    filename: str,
    code: str,
    chunk_size: int = 512,
    start_line: int | None = None,
) -> list[ParsedChunk]:
    """Split large code blocks into smaller code chunks.

    The function preserves line boundaries and creates sequential
    code chunks that do not exceed the configured chunk size (default=512 characters).

    Args:
        filename (str):
            Name of the source file.

        code (str):
            Source code content.

        chunk_size (int, optional):
            Maximum size of each chunk in characters.
            Defaults to 512.

        start_line (int, optional):
            1-based line the given ``code`` starts on in the source file, if
            known. Every chunk produced from ``code`` is tagged with this
            same value, so it points at the enclosing symbol's definition
            rather than at each individual split's own offset.

    Returns:
        list[ParsedChunk]:
            Sequential code chunks with metadata and chunk indices.
    """
    lines: list[str] = code.splitlines()
    chunks_content: list[str] = []
    current_chunk_content: list[str] = []
    current_chunk_content_length = 0
    for line in lines:
        if (
            current_chunk_content_length + len(line) > chunk_size
            and current_chunk_content
        ):
            chunks_content.append("\n".join(current_chunk_content))
            current_chunk_content = []
            current_chunk_content_length = 0

        current_chunk_content.append(line)
        current_chunk_content_length += len(line) + 1  # + 1, because each added line
        # brings a line break (/n) with it

    if current_chunk_content:
        chunks_content.append("\n".join(current_chunk_content))

    total_chunks_amount: int = len(chunks_content)

    return [
        to_parsed_chunk(
            chunk_content,
            "code",
            filename,
            chunk_index,
            total_chunks_amount,
            start_line=start_line,
        )
        for chunk_index, chunk_content in enumerate(chunks_content)
    ]

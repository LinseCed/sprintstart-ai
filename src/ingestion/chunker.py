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

    Returns:
        ParsedChunk:
            Chunk instance with content, type and metadata.
    """
    return ParsedChunk(
        content=chunk_content,
        kind=kind,
        metadata={
            **build_metadata(Path(filename)),
            "chunk_index": str(chunk_index),
            "total_chunks": str(total_chunks_amount),
        },
    )


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

    raw_chunks_content: list[str] = []

    paragraphs: list[str] = [
        paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
    ]
    current_chunk_content: list[str] = []

    for paragraph in paragraphs:
        paragraph_length: int = len(paragraph)

        # hard split by character when paragraph itself exceeds chunk_size
        if paragraph_length > chunk_size:
            if current_chunk_content:
                raw_chunks_content.append("\n\n".join(current_chunk_content))
                overlap_paragraph: str = (
                    current_chunk_content[-1]
                    if len(current_chunk_content[-1]) < chunk_overlap
                    else ""
                )
                current_chunk_content = [overlap_paragraph] if overlap_paragraph else []

            paragraph_with_overlap: str = "\n\n".join(
                current_chunk_content + [paragraph]
            )
            start: int = 0
            while start < len(paragraph_with_overlap):
                raw_chunks_content.append(
                    paragraph_with_overlap[start : start + chunk_size]
                )
                start += chunk_size - chunk_overlap

            continue

        # Would adding the paragraph exceed chunk_size?
        candidate_length: int = len("\n\n".join(current_chunk_content + [paragraph]))

        # handle when current_chunk_content + paragraph exceeds chunk size
        if current_chunk_content and (candidate_length > chunk_size):
            raw_chunks_content.append("\n\n".join(current_chunk_content))
            overlap_paragraph: str = current_chunk_content[-1]
            current_chunk_content = [overlap_paragraph]

        # append whole paragraph to current_chunk_content
        current_chunk_content.append(paragraph)

    # if there is some content left, append it to the raw_chunks
    if current_chunk_content:
        raw_chunks_content.append("\n\n".join(current_chunk_content))

    total_chunks_amount = len(raw_chunks_content)

    return [
        to_parsed_chunk(
            chunk_content, "text", filename, chunk_index, total_chunks_amount
        )
        for chunk_index, chunk_content in enumerate(raw_chunks_content)
    ]


def chunk_code(filename: str, code: str, chunk_size: int = 512) -> list[ParsedChunk]:
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
            chunk_content, "code", filename, chunk_index, total_chunks_amount
        )
        for chunk_index, chunk_content in enumerate(chunks_content)
    ]

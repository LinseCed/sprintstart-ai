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

def toParsedChunk(
    chunk_content: str,
    kind: ChunkKind,
    filename: str,
    chunk_index: int,
    total_chunks_amount: int,
):
    return ParsedChunk(
        content=chunk_content,
        kind=kind,
        metadata={
            **build_metadata(Path(filename)),
            "chunk_index": str(chunk_index),
            "total_chunks": str(total_chunks_amount),
        },
    )

def flush_chunk(
    parts: list[str],
    chunk_overlap: int,
) -> tuple[str, str, int]:
    chunk_content = "\n\n".join(parts)
    overlap: str = chunk_content[-chunk_overlap:] if chunk_overlap > 0 else ""
    return chunk_content, overlap, len(overlap)


def chunk_text(
    filename: str,
    text: str,
    chunk_size: int = chunk_size,
    chunk_overlap: int = chunk_overlap,
) -> list[ParsedChunk]:
    
    raw_chunks_content: list[str] = []

    paragraphs: list[str] = [
        paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
    ]
    current_chunk_content: list[str] = []
    current_chunk_content_length: int = 0

    for paragraph in paragraphs:
        paragraph_length: int = len(paragraph)

        # hard split when paragraph itself exceeds chunk_size
        if paragraph_length > chunk_size:
            if current_chunk_content:
                chunk_content, overlap, overlap_len = flush_chunk(
                    current_chunk_content,
                    chunk_overlap,
                ) 
                
                raw_chunks_content.append(chunk_content)
                current_chunk_content = [overlap] if overlap else []
                current_chunk_content_length = overlap_len

            paragraph_with_overlap: str = "\n\n".join(current_chunk_content) + paragraph
            start: int = 0
            while start < len(paragraph_with_overlap):
                paragraph_part: str = paragraph_with_overlap[start : start + chunk_size]
                chunk_content, overlap, overlap_len = flush_chunk(
                    [paragraph_part],
                    chunk_overlap,
                )
                raw_chunks_content.append(chunk_content)
                current_chunk_content = [overlap] if overlap else []
                current_chunk_content_length = overlap_len

                start += chunk_size - chunk_overlap
            continue


        # handle when current_chunk_content + paragraph exceeds chunk size
        if current_chunk_content and (
            current_chunk_content_length + paragraph_length > chunk_size
        ):
            chunk_content, overlap, overlap_len = flush_chunk(
                current_chunk_content,
                chunk_overlap,
            )
            
            raw_chunks_content.append(chunk_content)

            current_chunk_content = [overlap] if overlap else []
            current_chunk_content_length = overlap_len

        # append whole paragraph to current_chunk_content
        current_chunk_content.append(paragraph)
        current_chunk_content_length += paragraph_length

    # if there is some content left, append it to the raw_chunks
    if current_chunk_content:
        raw_chunks_content.append("\n\n".join(current_chunk_content))

    total_chunks_amount = len(raw_chunks_content)

    return [
        toParsedChunk(chunk_content, "text", filename, chunk_index, total_chunks_amount)
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
        toParsedChunk(chunk_content, "code", filename, chunk_index, total_chunks_amount)
        for chunk_index, chunk_content in enumerate(chunks_content)
    ]



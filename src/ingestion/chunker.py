from pathlib import Path

from ingestion.models import ParsedChunk
from ingestion.utils import build_metadata


def chunk_text(filename: str, text: str, chunk_size: int = 512) -> list[ParsedChunk]:
    """Split a text into fixed-size ParsedChunk objects.

    The function divides the input text into sequential chunks of
    at most `chunk_size` characters while preserving the original
    order of the content.

    Each generated chunk contains:
    - the chunk text content
    - the chunk type ("text")
    - metadata about the source file
    - a sequential chunk index


    Args:
        filename (str): Name of the source file.
        text (str): Decoded text content to split into chunks.
        chunk_size (int, optional):  Maximum number of characters
                                     per chunk. Defaults to 512.

    Returns:
        list[ParsedChunk]: A list of ParsedChunk objects containing
                           sequential text chunks with metadata.
    """

    chunks: list[ParsedChunk] = []

    for i in range(0, len(text), chunk_size):
        chunk_sized_text: str = text[i : i + chunk_size]

        chunks.append(
            ParsedChunk(
                content=chunk_sized_text,
                kind="text",
                metadata={
                    # unpack the dict
                    **build_metadata(Path(filename)),
                    "chunk_index": str(i // chunk_size),
                },
            )
        )

    return chunks


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
    chunk_count: int = 0
    chunks_content: dict[int, str] = {}
    current_chunk_content: list[str] = []
    current_chunk_content_length = 0
    for line in lines:
        if (
            current_chunk_content_length + len(line) > chunk_size
            and current_chunk_content
        ):
            chunks_content[chunk_count] = "\n".join(current_chunk_content)
            chunk_count += 1
            current_chunk_content = []
            current_chunk_content_length = 0

        current_chunk_content.append(line)
        current_chunk_content_length += len(line) + 1  # + 1, because each added line
        # brings a line break (/n) with it

    if current_chunk_content:
        chunks_content[chunk_count] = "\n".join(current_chunk_content)

    return [
        ParsedChunk(
            content=chunk_content,
            kind="code",
            metadata={
                **build_metadata(Path(filename)),
                "chunk_index": str(chunk_index),
                "total_chunks": str(chunk_count),
            },
        )
        for chunk_index, chunk_content in chunks_content.items()
    ]

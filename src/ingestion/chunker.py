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

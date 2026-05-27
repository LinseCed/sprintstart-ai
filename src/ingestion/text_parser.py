
from ingestion.chunker import chunk_text
from ingestion.models import ParsedChunk


def parse_text(filename: str, content: bytes) -> list[ParsedChunk]:
    """Parse plain text-based files into a single text chunk.

    Args:
        filename (str): Name of the source file.
        content (bytes): Raw file content as bytes.

    Returns:
        list[ParsedChunk]: A list containing a single ParsedChunk with kind='text'.
    """
    text = content.decode(encoding = "utf-8", errors = "replace")

    return chunk_text(filename, text)
from pathlib import Path

from ingestion.models import ParsedChunk
from ingestion.parser import _meta


def parse_text(filename: str, content: bytes) -> list[ParsedChunk]:
    """Parse plain text-based files into a single text chunk.

    Args:
        filename (str): Name of the source file.
        content (bytes): Raw file content as bytes.

    Returns:
        list[ParsedChunk]: A list containing a single ParsedChunk with kind='text'.
    """
    text = content.decode(encoding = "utf-8", errors = "replace")

    return [
        ParsedChunk(
            content = text,
            kind = "text",
            metadata = _meta(Path(filename)),
        )
    ]
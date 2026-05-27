
from pathlib import Path

from src.ingestion.models import ParsedChunk
from src.ingestion.utils import build_metadata


def chunk_text(filename: str, text: str, chunk_size: int = 512) -> list[ParsedChunk]:

    chunks: list[ParsedChunk] = []

    for i in range(0, len(text), chunk_size):
        chunk_sized_text: str = text[i: i + chunk_size] # TODO: IndexOutOfBoundsException??

        chunks.append(
            ParsedChunk(
                content = chunk_sized_text,
                kind = "text",
                metadata = {
                    # unpack the dict
                    **build_metadata(Path(filename)),
                    "chunk_index" : str(i//chunk_size),
                }
            )
        )

    return chunks
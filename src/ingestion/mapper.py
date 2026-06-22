import uuid

from ingestion.models import ParsedChunk
from rag.types import Chunk


def to_chunk(parsed: ParsedChunk, artifact_id: str, embedding: list[float]) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        artifact_id=artifact_id,
        filename=parsed.metadata["filename"],
        text=parsed.content,
        embedding=embedding,
        kind=parsed.kind,
        position=int(parsed.metadata.get("chunk_index", "0")),
    )

import hashlib

from ingestion.models import ParsedChunk
from ingestion.source_role import DEFAULT_SOURCE_ROLE, SourceRole
from rag.types import Chunk


def _deterministic_id(artifact_id: str, content: str, position: int) -> str:
    """Derive a stable chunk ID from its content and position.

    Using a deterministic ID means re-ingesting the same file produces the same
    IDs, so ChromaDB's ``upsert`` correctly deduplicates and the ``delete(...,
    exclude_ids=new_ids)`` cleanup pattern works.
    """
    h = hashlib.sha256(f"{artifact_id}:{position}:{content}".encode())
    return h.hexdigest()[:32]


def _optional_int(metadata: dict[str, str], key: str) -> int | None:
    raw = metadata.get(key)
    return int(raw) if raw is not None else None


def to_chunk(
    parsed: ParsedChunk,
    artifact_id: str,
    embedding: list[float],
    source_role: SourceRole = DEFAULT_SOURCE_ROLE,
) -> Chunk:
    position = int(parsed.metadata.get("chunk_index", "0"))
    return Chunk(
        id=_deterministic_id(artifact_id, parsed.content, position),
        artifact_id=artifact_id,
        filename=parsed.metadata["filename"],
        text=parsed.content,
        embedding=embedding,
        kind=parsed.kind,
        position=position,
        source_role=source_role,
        start_line=_optional_int(parsed.metadata, "start_line"),
        start_page=_optional_int(parsed.metadata, "page_number"),
    )

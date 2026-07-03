import uuid

from ingestion.models import ParsedChunk
from ingestion.source_role import SourceRole
from rag.types import Chunk, SourceType, is_source_type


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None

    return str(value)


def _source_type_for(
    parsed: ParsedChunk,
    artifact_type: str | None,
    language: str | None,
    source_type: SourceType | None,
) -> SourceType:
    if source_type is not None:
        return source_type

    raw_source_type = parsed.metadata.get("source_type")
    if raw_source_type is not None:
        parsed_source_type = str(raw_source_type)
        if is_source_type(parsed_source_type):
            return parsed_source_type

    artifact_type_value = (
        artifact_type or _optional_str(parsed.metadata.get("artifact_type")) or ""
    ).upper()

    language_value = language or _optional_str(parsed.metadata.get("language"))
    filename = str(parsed.metadata.get("filename", "")).lower()

    if parsed.kind == "code" or language_value:
        return "code"

    if artifact_type_value in {"ISSUE", "TICKET", "PULL_REQUEST"}:
        return "tickets"

    if "ticket" in filename or "issue" in filename:
        return "tickets"

    return "docs"


def _source_timestamp_for(
    parsed: ParsedChunk,
    source_created_at: str | None,
    source_updated_at: str | None,
) -> str | None:
    return (
        source_updated_at
        or source_created_at
        or _optional_str(parsed.metadata.get("source_updated_at"))
        or _optional_str(parsed.metadata.get("source_created_at"))
        or _optional_str(parsed.metadata.get("updated_at"))
        or _optional_str(parsed.metadata.get("created_at"))
        or _optional_str(parsed.metadata.get("indexed_at"))
    )


def to_chunk(
    parsed: ParsedChunk,
    artifact_id: str,
    embedding: list[float],
    source_role: SourceRole = "primary",
    source_url: str | None = None,
    artifact_type: str | None = None,
    language: str | None = None,
    source_created_at: str | None = None,
    source_updated_at: str | None = None,
    source_type: SourceType | None = None,
) -> Chunk:
    effective_source_url = source_url or _optional_str(
        parsed.metadata.get("source_url")
    )
    effective_artifact_type = artifact_type or _optional_str(
        parsed.metadata.get("artifact_type")
    )
    effective_language = language or _optional_str(parsed.metadata.get("language"))

    return Chunk(
        id=str(uuid.uuid4()),
        artifact_id=artifact_id,
        filename=str(parsed.metadata["filename"]),
        text=parsed.content,
        embedding=embedding,
        kind=parsed.kind,
        position=int(parsed.metadata.get("chunk_index", "0")),
        heading_path=None,
        source_role=source_role,
        source_url=effective_source_url,
        artifact_type=effective_artifact_type,
        language=effective_language,
        source_type=_source_type_for(
            parsed,
            artifact_type=effective_artifact_type,
            language=effective_language,
            source_type=source_type,
        ),
        created_at=_source_timestamp_for(
            parsed,
            source_created_at=source_created_at,
            source_updated_at=source_updated_at,
        ),
    )

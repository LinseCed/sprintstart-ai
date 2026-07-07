from dataclasses import dataclass
from typing import TypeGuard

from ingestion.models import ChunkKind
from ingestion.source_role import DEFAULT_SOURCE_ROLE, SourceRole


def is_chunk_kind(value: str) -> TypeGuard[ChunkKind]:
    return value in ("text", "code", "pdf", "image")


@dataclass(frozen=True)
class Chunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    embedding: list[float]
    position: int | None = None
    kind: ChunkKind = "text"
    source_role: SourceRole = DEFAULT_SOURCE_ROLE
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    score: float
    position: int | None = None
    kind: ChunkKind = "text"
    source_role: SourceRole = DEFAULT_SOURCE_ROLE
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class Citation:
    filename: str
    chunk_id: str
    artifact_id: str
    source_url: str | None = None

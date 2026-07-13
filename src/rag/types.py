from dataclasses import dataclass
from typing import Literal, TypeGuard

from ingestion.source_role import SourceRole

ChunkKind = Literal["text", "code", "pdf", "image"]
SourceSystem = Literal["GITHUB", "JIRA", "UPLOAD"]


def is_chunk_kind(value: str) -> TypeGuard[ChunkKind]:
    return value in ("text", "code", "pdf", "image")


def is_source_system(value: str) -> TypeGuard[SourceSystem]:
    return value in ("GITHUB", "JIRA", "UPLOAD")


@dataclass(frozen=True)
class RetrievalFilters:
    source_systems: list[SourceSystem] | None = None
    time_from: str | None = None
    time_to: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    embedding: list[float]
    kind: ChunkKind = "text"
    position: int | None = None
    source_role: SourceRole = "primary"
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None
    source_system: SourceSystem | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    score: float
    kind: ChunkKind = "text"
    position: int | None = None
    source_role: SourceRole = "primary"
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None
    source_system: SourceSystem | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    filename: str
    source_url: str | None = None

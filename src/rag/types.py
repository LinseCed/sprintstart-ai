from dataclasses import dataclass
from typing import Literal, TypeGuard

from ingestion.models import ChunkKind
from ingestion.source_role import SourceRole

SourceType = Literal["docs", "code", "tickets"]
TimeRange = Literal["latest", "last_6_months"]


def is_chunk_kind(value: str) -> TypeGuard[ChunkKind]:
    return value in ("text", "code", "pdf")


def is_source_type(value: str) -> TypeGuard[SourceType]:
    return value in ("docs", "code", "tickets")


@dataclass(frozen=True)
class RetrievalFilters:
    source_type: SourceType | None = None
    time_range: TimeRange | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    embedding: list[float]
    heading_path: str | None = None
    position: int | None = None
    kind: ChunkKind = "text"
    source_role: SourceRole = "primary"
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None
    source_type: SourceType | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    score: float
    heading_path: str | None = None
    position: int | None = None
    kind: ChunkKind = "text"
    source_role: SourceRole = "primary"
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None
    source_type: SourceType | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class Citation:
    filename: str
    section_path: str | None
    chunk_id: str
    source_url: str | None = None

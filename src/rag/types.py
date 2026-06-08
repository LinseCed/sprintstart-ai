from dataclasses import dataclass
from typing import TypeGuard

from ingestion.models import ChunkKind


def is_chunk_kind(value: str) -> TypeGuard[ChunkKind]:
    return value in ("text", "code", "pdf", "image")


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


@dataclass(frozen=True)
class Citation:
    filename: str
    section_path: str | None
    chunk_id: str

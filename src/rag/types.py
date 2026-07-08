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
    # 1-based line the chunk starts on in the source file. Only meaningful for
    # "text"/"code" chunks; PDFs track the source page instead (``start_page``).
    start_line: int | None = None
    # 1-based PDF page the chunk was extracted from. Only meaningful for "pdf"
    # chunks; text/code chunks track the source line instead (``start_line``).
    start_page: int | None = None


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
    start_line: int | None = None
    start_page: int | None = None


@dataclass(frozen=True)
class Citation:
    filename: str
    chunk_id: str
    artifact_id: str
    source_url: str | None = None
    start_line: int | None = None
    start_page: int | None = None

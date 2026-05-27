from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Chunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    embedding: list[float]
    heading_path: str | None = None
    position: int | None = None
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    score: float
    heading_path: str | None = None
    position: int | None = None
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class Citation:
    filename: str
    section_path: str | None
    chunk_id: str

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    embedding: list[float]
    heading_path: str | None = None
    position: int | None = None
    kind: str = "text"
    score: float | None = None


@dataclass(frozen=True)
class Citation:
    filename: str
    section_path: str | None
    chunk_id: str

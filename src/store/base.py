from typing import Protocol

from rag.types import Chunk, ScoredChunk


class VectorStore(Protocol):

    def add(self, chunks: list[Chunk]) -> None:
        ...

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[ScoredChunk]:
        ...

    def delete(self, artifact_id: str) -> None:
        ...

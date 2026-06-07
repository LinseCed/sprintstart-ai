from typing import Protocol

from rag.types import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[Chunk]: ...

    def delete(self, artifact_id: str) -> None: ...

    def all_chunks(self) -> list[Chunk]: ...

    def count(self) -> int: ...

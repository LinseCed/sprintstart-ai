from rag.types import Chunk, ScoredChunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class StubVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        new_ids = {c.id for c in chunks}
        self.chunks = [c for c in self.chunks if c.id not in new_ids] + chunks

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(
                id=chunk.id,
                artifact_id=chunk.artifact_id,
                filename=chunk.filename,
                position=chunk.position,
                kind=chunk.kind,
                text=chunk.text,
                score=cosine_similarity(embedding, chunk.embedding),
                source_role=chunk.source_role,
                source_url=chunk.source_url,
                artifact_type=chunk.artifact_type,
                language=chunk.language,
            )
            for chunk in self.chunks
        ]

        return [
            chunk
            for chunk in sorted(
                scored,
                key=lambda item: item.score,
                reverse=True,
            )[:top_k]
            if chunk.score >= min_score
        ]

    def delete(self, artifact_id: str, exclude_ids: list[str] | None = None) -> int:
        before = len(self.chunks)
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.artifact_id != artifact_id or chunk.id in (exclude_ids or [])
        ]
        return before - len(self.chunks)

    def list_chunks(self, limit: int, offset: int = 0) -> list[Chunk]:
        return list(self.chunks[offset : offset + limit])

    def list_chunks_by_artifact(
        self,
        artifact_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Chunk]:
        matching = [c for c in self.chunks if c.artifact_id == artifact_id]
        return matching[offset : offset + limit]

    def count_by_artifact(self, artifact_id: str) -> int:
        return sum(1 for c in self.chunks if c.artifact_id == artifact_id)

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)

    def count(self) -> int:
        return len(self.chunks)

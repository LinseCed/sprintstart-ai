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
        self.chunks = self.chunks + chunks

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
                heading_path=chunk.heading_path,
                position=chunk.position,
                kind=chunk.kind,
                text=chunk.text,
                score=cosine_similarity(embedding, chunk.embedding),
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

    def delete(self, artifact_id: str, exclude_ids: list[str] | None = None) -> None:
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.artifact_id != artifact_id or chunk.id in (exclude_ids or [])
        ]

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)

    def count(self) -> int:
        return len(self.chunks)

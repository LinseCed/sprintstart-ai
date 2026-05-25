from src.rag.types import Chunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class StubVectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[Chunk]:
        scored = [
            Chunk(
                id=chunk.id,
                artifact_id=chunk.artifact_id,
                filename=chunk.filename,
                heading_path=chunk.heading_path,
                position=chunk.position,
                kind=chunk.kind,
                text=chunk.text,
                embedding=chunk.embedding,
                score=cosine_similarity(embedding, chunk.embedding),
            )
            for chunk in self.chunks
        ]

        return [
            chunk
            for chunk in sorted(
                scored,
                key=lambda item: item.score or 0.0,
                reverse=True,
            )[:top_k]
            if (chunk.score or 0.0) >= min_score
        ]

    def delete(self, artifact_id: str) -> None:
        self.chunks = [
            chunk for chunk in self.chunks if chunk.artifact_id != artifact_id
        ]

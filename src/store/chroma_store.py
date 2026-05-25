from typing import Any

import chromadb

from src.rag.types import Chunk


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "chunks",
        client: Any | None = None,
        path: str | None = None,
    ) -> None:
        if client is not None:
            self.client = client
        elif path is not None:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[
                {
                    "artifact_id": chunk.artifact_id,
                    "filename": chunk.filename,
                    "heading_path": chunk.heading_path or "",
                    "position": chunk.position if chunk.position is not None else -1,
                    "kind": chunk.kind,
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[Chunk]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[Chunk] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            score = 1 - distance

            if score < min_score:
                continue

            raw_position = metadata.get("position")
            position = None if raw_position == -1 else raw_position

            chunks.append(
                Chunk(
                    id=chunk_id,
                    artifact_id=metadata["artifact_id"],
                    filename=metadata["filename"],
                    heading_path=metadata.get("heading_path") or None,
                    position=position,
                    kind=metadata.get("kind", "text"),
                    text=text,
                    embedding=[],
                    score=score,
                )
            )

        chunks.sort(key=lambda chunk: chunk.score or 0.0, reverse=True)
        return chunks

    def delete(self, artifact_id: str) -> None:
        result = self.collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )

        ids = result.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)

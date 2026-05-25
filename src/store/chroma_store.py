from typing import Any, cast

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
        raw_result = cast(
            dict[str, Any],
            self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            ),
        )

        ids = cast(list[list[str]], raw_result.get("ids") or [[]])[0]
        documents = cast(list[list[str]], raw_result.get("documents") or [[]])[0]
        metadatas = cast(
            list[list[dict[str, Any]]],
            raw_result.get("metadatas") or [[]],
        )[0]
        distances = cast(list[list[float]], raw_result.get("distances") or [[]])[0]

        chunks: list[Chunk] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            score = 1.0 - distance

            if score < min_score:
                continue

            raw_heading_path = metadata.get("heading_path")
            heading_path = str(raw_heading_path) if raw_heading_path else None

            raw_position = metadata.get("position")
            position = None if raw_position in (None, -1) else int(raw_position)

            raw_kind = metadata.get("kind", "text")

            chunks.append(
                Chunk(
                    id=str(chunk_id),
                    artifact_id=str(metadata["artifact_id"]),
                    filename=str(metadata["filename"]),
                    heading_path=heading_path,
                    position=position,
                    kind=str(raw_kind),
                    text=str(text),
                    embedding=[],
                    score=score,
                )
            )

        chunks.sort(key=lambda chunk: chunk.score or 0.0, reverse=True)
        return chunks

    def delete(self, artifact_id: str) -> None:
        raw_result = cast(
            dict[str, Any],
            self.collection.get(
                where={"artifact_id": artifact_id},
                include=[],
            ),
        )

        ids = cast(list[str], raw_result.get("ids") or [])

        if ids:
            self.collection.delete(ids=ids)

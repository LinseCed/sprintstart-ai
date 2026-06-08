from typing import cast

import chromadb
import chromadb.api
from chromadb.api.types import PyEmbeddings

from rag.types import Chunk, ScoredChunk, is_chunk_kind

_NO_POSITION: int = -1


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "chunks",
        client: chromadb.api.ClientAPI | None = None,
        path: str | None = None,
    ) -> None:
        if client is not None:
            self._client: chromadb.api.ClientAPI = client
        elif path is not None:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        embeddings: list[list[float]] = [chunk.embedding for chunk in chunks]

        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=cast(PyEmbeddings, embeddings),
            metadatas=[
                {
                    "artifact_id": chunk.artifact_id,
                    "filename": chunk.filename,
                    "heading_path": chunk.heading_path or "",
                    "position": (
                        chunk.position if chunk.position is not None else _NO_POSITION
                    ),
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
    ) -> list[ScoredChunk]:
        if self._collection.count() == 0:
            return []

        raw_result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw_result["ids"][0]
        documents = (raw_result["documents"] or [[]])[0]
        metadatas = (raw_result["metadatas"] or [[]])[0]
        distances = (raw_result["distances"] or [[]])[0]

        results: list[ScoredChunk] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            score = 1.0 - distance

            if score < min_score:
                continue

            raw_heading_path = metadata.get("heading_path")
            heading_path = str(raw_heading_path) if raw_heading_path else None

            raw_position = metadata.get("position")
            position = (
                None
                if not isinstance(raw_position, (int, float))
                or raw_position == _NO_POSITION
                else int(raw_position)
            )

            kind_str = str(metadata.get("kind", "text"))
            if not is_chunk_kind(kind_str):
                raise ValueError(f"Unknown chunk kind {kind_str!r}")

            results.append(
                ScoredChunk(
                    id=str(chunk_id),
                    artifact_id=str(metadata["artifact_id"]),
                    filename=str(metadata["filename"]),
                    heading_path=heading_path,
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    score=score,
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return results

    def delete(self, artifact_id: str, exclude_ids: list[str] | None = None) -> None:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )

        ids = raw_result["ids"]

        if exclude_ids:
            ids = [i for i in ids if i not in exclude_ids]

        if ids:
            self._collection.delete(ids=ids)

    def all_chunks(self) -> list[Chunk]:
        raw_result = self._collection.get(
            include=["documents", "metadatas", "embeddings"],
        )

        ids = raw_result["ids"]
        documents = raw_result["documents"] or []
        metadatas = raw_result["metadatas"] or []
        embeddings = raw_result["embeddings"] if raw_result["embeddings"] is not None else []

        chunks: list[Chunk] = []

        for chunk_id, text, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
            raw_heading_path = metadata.get("heading_path")
            heading_path = str(raw_heading_path) if raw_heading_path else None

            raw_position = metadata.get("position")
            position = (
                None
                if not isinstance(raw_position, (int, float))
                or raw_position == _NO_POSITION
                else int(raw_position)
            )

            kind_str = str(metadata.get("kind", "text"))
            if not is_chunk_kind(kind_str):
                raise ValueError(f"Unknown chunk kind {kind_str!r}")

            chunks.append(
                Chunk(
                    id=str(chunk_id),
                    artifact_id=str(metadata["artifact_id"]),
                    filename=str(metadata["filename"]),
                    heading_path=heading_path,
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    embedding=list(embedding),
                )
            )

        return chunks

    def count(self) -> int:
        return self._collection.count()

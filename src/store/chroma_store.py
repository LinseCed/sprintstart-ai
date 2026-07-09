from collections.abc import Mapping
from typing import cast

import chromadb
import chromadb.api
from chromadb.api.types import PyEmbeddings

from ingestion.source_role import DEFAULT_SOURCE_ROLE, SourceRole, is_source_role
from rag.types import Chunk, ScoredChunk, is_chunk_kind

_NO_POSITION: int = -1


def _optional_str(metadata: Mapping[str, object], key: str) -> str | None:
    """Return the metadata value as a non-empty string, or None."""
    raw = metadata.get(key)
    return str(raw) if raw else None


def _source_role_of(metadata: Mapping[str, object]) -> SourceRole:
    """Read the source role from chunk metadata.

    Legacy chunks ingested before roles existed have no ``source_role`` key;
    they default to ``primary`` so existing corpora keep behaving as before.
    """
    raw = str(metadata.get("source_role", DEFAULT_SOURCE_ROLE))
    return raw if is_source_role(raw) else DEFAULT_SOURCE_ROLE


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
                    "position": (
                        chunk.position if chunk.position is not None else _NO_POSITION
                    ),
                    "kind": chunk.kind,
                    "source_role": chunk.source_role,
                    "source_url": chunk.source_url or "",
                    "artifact_type": chunk.artifact_type or "",
                    "language": chunk.language or "",
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
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    score=score,
                    source_role=_source_role_of(metadata),
                    source_url=_optional_str(metadata, "source_url"),
                    artifact_type=_optional_str(metadata, "artifact_type"),
                    language=_optional_str(metadata, "language"),
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return results

    def delete(
        self,
        artifact_id: str,
        exclude_ids: list[str] | None = None,
    ) -> int:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )

        ids = raw_result["ids"]

        if exclude_ids:
            ids = [chunk_id for chunk_id in ids if chunk_id not in exclude_ids]

        deleted_count = len(ids)

        if ids:
            self._collection.delete(ids=ids)

        return deleted_count

    def list_chunks(self, limit: int, offset: int = 0) -> list[Chunk]:
        raw_result = self._collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=limit,
            offset=offset,
        )

        ids = raw_result["ids"]
        documents = raw_result["documents"] or []
        metadatas = raw_result["metadatas"] or []
        embeddings = (
            raw_result["embeddings"] if raw_result["embeddings"] is not None else []
        )

        chunks: list[Chunk] = []

        for chunk_id, text, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
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
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    embedding=list(embedding),
                    source_role=_source_role_of(metadata),
                    source_url=_optional_str(metadata, "source_url"),
                    artifact_type=_optional_str(metadata, "artifact_type"),
                    language=_optional_str(metadata, "language"),
                )
            )

        return chunks

    def list_chunks_by_artifact(
        self,
        artifact_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Chunk]:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=["documents", "metadatas", "embeddings"],
            limit=limit,
            offset=offset,
        )

        ids = raw_result["ids"]
        documents = raw_result["documents"] or []
        metadatas = raw_result["metadatas"] or []
        embeddings = (
            raw_result["embeddings"] if raw_result["embeddings"] is not None else []
        )

        chunks: list[Chunk] = []

        for chunk_id, text, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
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
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    embedding=list(embedding),
                    source_role=_source_role_of(metadata),
                    source_url=_optional_str(metadata, "source_url"),
                    artifact_type=_optional_str(metadata, "artifact_type"),
                    language=_optional_str(metadata, "language"),
                )
            )

        return chunks

    def count_by_artifact(self, artifact_id: str) -> int:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )
        return len(raw_result["ids"])

    def all_chunks(self) -> list[Chunk]:
        total = self.count()

        if total == 0:
            return []

        return self.list_chunks(limit=total, offset=0)

    def all_chunks_without_embeddings(self) -> list[Chunk]:
        total = self.count()
        if total == 0:
            return []
        return self.list_chunks_without_embeddings(limit=total, offset=0)

    def list_chunks_without_embeddings(
        self, limit: int, offset: int = 0
    ) -> list[Chunk]:
        raw_result = self._collection.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )

        ids = raw_result["ids"]
        documents = raw_result["documents"] or []
        metadatas = raw_result["metadatas"] or []

        chunks: list[Chunk] = []

        for chunk_id, text, metadata in zip(
            ids,
            documents,
            metadatas,
            strict=True,
        ):
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
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    embedding=[],  # no embeddings in text-only fetch
                    source_role=_source_role_of(metadata),
                    source_url=_optional_str(metadata, "source_url"),
                    artifact_type=_optional_str(metadata, "artifact_type"),
                    language=_optional_str(metadata, "language"),
                )
            )

        return chunks

    def all_ids(self) -> frozenset[str]:
        raw_result = self._collection.get(include=[])
        return frozenset(str(chunk_id) for chunk_id in raw_result["ids"])

    def count(self) -> int:
        return self._collection.count()

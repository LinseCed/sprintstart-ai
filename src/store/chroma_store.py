from collections.abc import Mapping
from typing import Any, cast

import chromadb
import chromadb.api
from chromadb.api.types import PyEmbeddings

from ingestion.source_role import SourceRole
from rag.filters import (
    normalize_source_system,
    timestamp_from_iso,
    where_filter_for_chroma,
)
from rag.types import Chunk, RetrievalFilters, ScoredChunk, is_chunk_kind

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
                    "position": (
                        chunk.position if chunk.position is not None else _NO_POSITION
                    ),
                    "kind": chunk.kind,
                    "source_role": chunk.source_role,
                    "source_url": chunk.source_url or "",
                    "artifact_type": chunk.artifact_type or "",
                    "language": chunk.language or "",
                    "source_system": chunk.source_system or "",
                    "created_at": chunk.created_at or "",
                    "created_at_ts": timestamp_from_iso(chunk.created_at),
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
        filters: RetrievalFilters | None = None,
    ) -> list[ScoredChunk]:
        if self._collection.count() == 0:
            return []

        where_filter = where_filter_for_chroma(filters)

        raw_result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw_result["ids"][0]
        documents = (raw_result["documents"] or [[]])[0]
        metadatas = cast(
            list[Mapping[str, object]],
            (raw_result["metadatas"] or [[]])[0],
        )
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

            source_system = normalize_source_system(
                _optional_str(metadata.get("source_system"))
            )

            results.append(
                ScoredChunk(
                    id=str(chunk_id),
                    artifact_id=str(metadata["artifact_id"]),
                    filename=str(metadata["filename"]),
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    score=score,
                    source_role=_source_role_from_metadata(metadata),
                    source_url=_optional_str(metadata.get("source_url")),
                    artifact_type=_optional_str(metadata.get("artifact_type")),
                    language=_optional_str(metadata.get("language")),
                    source_system=source_system,
                    created_at=_optional_str(metadata.get("created_at")),
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return results

    def delete(self, artifact_id: str, exclude_ids: list[str] | None = None) -> int:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )

        ids = list(raw_result["ids"])

        if exclude_ids:
            ids = [i for i in ids if i not in exclude_ids]

        if ids:
            self._collection.delete(ids=ids)

        return len(ids)

    def list_chunks(self, limit: int, offset: int = 0) -> list[Chunk]:
        raw_result = self._collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        return _chunks_from_get_result(raw_result)

    def list_chunks_by_artifact(
        self,
        artifact_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Chunk]:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        return _chunks_from_get_result(raw_result)

    def count_by_artifact(self, artifact_id: str) -> int:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )
        return len(raw_result["ids"])

    def all_chunks(self) -> list[Chunk]:
        raw_result = self._collection.get(
            include=["documents", "metadatas", "embeddings"],
        )
        return _chunks_from_get_result(raw_result)

    def count(self) -> int:
        return self._collection.count()


def _chunks_from_get_result(raw_result: Any) -> list[Chunk]:
    ids = cast(list[str], raw_result["ids"])
    documents = cast(list[str], raw_result.get("documents") or [])
    metadatas = cast(list[Mapping[str, object]], raw_result.get("metadatas") or [])
    raw_embeddings = raw_result.get("embeddings")
    if raw_embeddings is None:
        embeddings: list[list[float]] = []
    elif hasattr(raw_embeddings, "tolist"):
        embeddings = cast(list[list[float]], raw_embeddings.tolist())
    else:
        embeddings = cast(list[list[float]], raw_embeddings)

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

        source_system = normalize_source_system(
            _optional_str(metadata.get("source_system"))
        )

        chunks.append(
            Chunk(
                id=str(chunk_id),
                artifact_id=str(metadata["artifact_id"]),
                filename=str(metadata["filename"]),
                position=position,
                kind=kind_str,
                text=str(text),
                embedding=[float(value) for value in embedding],
                source_role=_source_role_from_metadata(metadata),
                source_url=_optional_str(metadata.get("source_url")),
                artifact_type=_optional_str(metadata.get("artifact_type")),
                language=_optional_str(metadata.get("language")),
                source_system=source_system,
                created_at=_optional_str(metadata.get("created_at")),
            )
        )

    return chunks


def _source_role_from_metadata(metadata: Mapping[str, object]) -> SourceRole:
    raw_source_role = metadata.get("source_role")
    if raw_source_role in {"primary", "test"}:
        return cast(SourceRole, raw_source_role)

    return "primary"


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None

    return str(value)

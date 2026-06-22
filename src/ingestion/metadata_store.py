from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

IngestionStatus = Literal["processing", "completed", "failed"]


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    filename: str
    content_type: str
    source_type: str
    size_bytes: int
    chunk_count: int
    status: IngestionStatus
    created_at: str
    updated_at: str
    error_message: str | None = None


@dataclass(frozen=True)
class ArtifactChunkRecord:
    id: str
    artifact_id: str
    filename: str
    text: str
    heading_path: list[str]
    chunk_index: int
    vector_store_id: str
    kind: str
    created_at: str


class IngestionMetadataStore:
    def __init__(self, path: str) -> None:
        self.path = path

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifact_chunks (
                    id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    text TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    vector_store_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
                )
                """
            )

    def save_artifact(self, artifact: ArtifactRecord) -> None:
        with self._connect() as connection:
            self._upsert_artifact(connection, artifact)

    def save_completed_artifact(
        self,
        artifact: ArtifactRecord,
        chunks: list[ArtifactChunkRecord],
    ) -> None:
        with self._connect() as connection:
            self._upsert_artifact(connection, artifact)

            connection.execute(
                "DELETE FROM artifact_chunks WHERE artifact_id = ?",
                (artifact.id,),
            )

            connection.executemany(
                """
                INSERT INTO artifact_chunks (
                    id,
                    artifact_id,
                    filename,
                    text,
                    heading_path,
                    chunk_index,
                    vector_store_id,
                    kind,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.artifact_id,
                        chunk.filename,
                        chunk.text,
                        json.dumps(chunk.heading_path),
                        chunk.chunk_index,
                        chunk.vector_store_id,
                        chunk.kind,
                        chunk.created_at,
                    )
                    for chunk in chunks
                ],
            )

    def mark_failed(
        self,
        artifact_id: str,
        error_message: str,
        updated_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE artifacts
                SET status = ?,
                    updated_at = ?,
                    error_message = ?
                WHERE id = ?
                """,
                ("failed", updated_at, error_message, artifact_id),
            )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT
                    id,
                    filename,
                    content_type,
                    source_type,
                    size_bytes,
                    chunk_count,
                    status,
                    created_at,
                    updated_at,
                    error_message
                FROM artifacts
                WHERE id = ?
                """,
                (artifact_id,),
            )
            row = cast(sqlite3.Row | None, cursor.fetchone())

        if row is None:
            return None

        return ArtifactRecord(
            id=str(row["id"]),
            filename=str(row["filename"]),
            content_type=str(row["content_type"]),
            source_type=str(row["source_type"]),
            size_bytes=int(row["size_bytes"]),
            chunk_count=int(row["chunk_count"]),
            status=cast(IngestionStatus, str(row["status"])),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            error_message=(
                None if row["error_message"] is None else str(row["error_message"])
            ),
        )

    def get_chunks(self, artifact_id: str) -> list[ArtifactChunkRecord]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT
                    id,
                    artifact_id,
                    filename,
                    text,
                    heading_path,
                    chunk_index,
                    vector_store_id,
                    kind,
                    created_at
                FROM artifact_chunks
                WHERE artifact_id = ?
                ORDER BY chunk_index ASC
                """,
                (artifact_id,),
            )
            rows = cast(list[sqlite3.Row], cursor.fetchall())

        return [self._chunk_from_row(row) for row in rows]

    def _upsert_artifact(
        self,
        connection: sqlite3.Connection,
        artifact: ArtifactRecord,
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                id,
                filename,
                content_type,
                source_type,
                size_bytes,
                chunk_count,
                status,
                created_at,
                updated_at,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.filename,
                artifact.content_type,
                artifact.source_type,
                artifact.size_bytes,
                artifact.chunk_count,
                artifact.status,
                artifact.created_at,
                artifact.updated_at,
                artifact.error_message,
            ),
        )

    def _decode_heading_path(self, heading_path_raw: str) -> list[str]:
        try:
            heading_path_data: object = json.loads(heading_path_raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(heading_path_data, list):
            return []

        typed_heading_path = cast(list[object], heading_path_data)
        return [str(item) for item in typed_heading_path]

    def _chunk_from_row(self, row: sqlite3.Row) -> ArtifactChunkRecord:
        return ArtifactChunkRecord(
            id=str(row["id"]),
            artifact_id=str(row["artifact_id"]),
            filename=str(row["filename"]),
            text=str(row["text"]),
            heading_path=self._decode_heading_path(str(row["heading_path"])),
            chunk_index=int(row["chunk_index"]),
            vector_store_id=str(row["vector_store_id"]),
            kind=str(row["kind"]),
            created_at=str(row["created_at"]),
        )

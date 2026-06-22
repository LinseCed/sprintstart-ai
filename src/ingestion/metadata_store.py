from __future__ import annotations

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
            self._ensure_artifact_chunks_schema(connection)

    def _ensure_artifact_chunks_schema(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        if not self._table_exists(connection, "artifact_chunks"):
            self._create_artifact_chunks_table(connection)
            return

        columns = self._table_columns(connection, "artifact_chunks")

        if "heading_path" not in columns:
            return

        connection.execute(
            "ALTER TABLE artifact_chunks RENAME TO artifact_chunks_legacy"
        )
        self._create_artifact_chunks_table(connection)

        legacy_columns = self._table_columns(connection, "artifact_chunks_legacy")
        columns_to_copy = [
            "id",
            "artifact_id",
            "filename",
            "text",
            "chunk_index",
            "vector_store_id",
            "kind",
            "created_at",
        ]

        if all(column in legacy_columns for column in columns_to_copy):
            columns_sql = ", ".join(columns_to_copy)
            connection.execute(
                f"""
                INSERT OR REPLACE INTO artifact_chunks ({columns_sql})
                SELECT {columns_sql}
                FROM artifact_chunks_legacy
                """
            )

        connection.execute("DROP TABLE artifact_chunks_legacy")

    def _table_exists(
        self,
        connection: sqlite3.Connection,
        table_name: str,
    ) -> bool:
        cursor = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        )
        return cursor.fetchone() is not None

    def _table_columns(
        self,
        connection: sqlite3.Connection,
        table_name: str,
    ) -> set[str]:
        cursor = connection.execute(f"PRAGMA table_info({table_name})")
        rows = cast(list[sqlite3.Row], cursor.fetchall())
        return {str(row["name"]) for row in rows}

    def _create_artifact_chunks_table(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            """
            CREATE TABLE artifact_chunks (
                id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                text TEXT NOT NULL,
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
                    chunk_index,
                    vector_store_id,
                    kind,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.artifact_id,
                        chunk.filename,
                        chunk.text,
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

    def _chunk_from_row(self, row: sqlite3.Row) -> ArtifactChunkRecord:
        return ArtifactChunkRecord(
            id=str(row["id"]),
            artifact_id=str(row["artifact_id"]),
            filename=str(row["filename"]),
            text=str(row["text"]),
            chunk_index=int(row["chunk_index"]),
            vector_store_id=str(row["vector_store_id"]),
            kind=str(row["kind"]),
            created_at=str(row["created_at"]),
        )

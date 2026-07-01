from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
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
    source_id: str | None = None
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None


class IngestionMetadataStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = RLock()

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._connection.execute(
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
                    error_message TEXT,
                    source_id TEXT,
                    source_url TEXT,
                    artifact_type TEXT,
                    language TEXT
                )
                """
            )

            self._connection.execute("DROP TABLE IF EXISTS artifact_chunks")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save_artifact(self, artifact: ArtifactRecord) -> None:
        with self._lock:
            self._upsert_artifact(artifact)
            self._connection.commit()

    def save_completed_artifact(self, artifact: ArtifactRecord) -> None:
        with self._lock:
            self._upsert_artifact(artifact)
            self._connection.commit()

    def mark_failed(
        self,
        artifact_id: str,
        error_message: str,
        updated_at: str,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE artifacts
                SET status = ?,
                    updated_at = ?,
                    error_message = ?
                WHERE id = ?
                """,
                ("failed", updated_at, error_message, artifact_id),
            )
            self._connection.commit()

    def mark_deindexed(self, artifact_id: str, updated_at: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE artifacts
                SET status = ?,
                    chunk_count = ?,
                    updated_at = ?,
                    error_message = ?
                WHERE id = ?
                """,
                ("deindexed", 0, updated_at, None, artifact_id),
            )
            self._connection.commit()

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        with self._lock:
            cursor = self._connection.execute(
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
                    error_message,
                    source_id,
                    source_url,
                    artifact_type,
                    language
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
            source_id=None if row["source_id"] is None else str(row["source_id"]),
            source_url=None if row["source_url"] is None else str(row["source_url"]),
            artifact_type=(
                None if row["artifact_type"] is None else str(row["artifact_type"])
            ),
            language=None if row["language"] is None else str(row["language"]),
        )

    def _upsert_artifact(self, artifact: ArtifactRecord) -> None:
        self._connection.execute(
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
                error_message,
                source_id,
                source_url,
                artifact_type,
                language
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                artifact.source_id,
                artifact.source_url,
                artifact.artifact_type,
                artifact.language,
            ),
        )

from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from ingestion.metadata_store import IngestionMetadataStore
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk, ScoredChunk


class StubVectorStore:
    def __init__(self) -> None:
        self.added_chunks: list[Chunk] = []
        self.deleted_artifacts: list[str] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.added_chunks.extend(chunks)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[ScoredChunk]:
        return []

    def delete(
        self,
        artifact_id: str,
        exclude_ids: list[str] | None = None,
    ) -> None:
        self.deleted_artifacts.append(artifact_id)

    def all_chunks(self) -> list[Chunk]:
        return self.added_chunks

    def count(self) -> int:
        return len(self.added_chunks)


class StubLLMClient:
    def generate(self, messages: list[Message]) -> str:
        return "answer"

    def stream(self, messages: list[Message]) -> Iterable[str]:
        yield "answer"

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def caption_image(self, image_bytes: bytes) -> str:
        return "caption"


class FailingEmbedLLMClient(StubLLMClient):
    def embed(self, text: str) -> list[float]:
        raise LLMUnavailableError("embedding backend unavailable")


@pytest.fixture
def vector_store() -> StubVectorStore:
    return StubVectorStore()


@pytest.fixture
def metadata_store(tmp_path: Path) -> IngestionMetadataStore:
    return IngestionMetadataStore(path=str(tmp_path / "metadata.db"))


@pytest.fixture
def client(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> Iterable[TestClient]:
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: StubLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_ingest_returns_artifact_and_chunk_metadata(
    client: TestClient,
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-1",
            "filename": "notes.txt",
            "content": "SprintStart uses OLLAMA_EMBED_MODEL for embeddings.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["artifact_id"] == "artifact-1"
    assert body["chunk_count"] >= 1
    assert body["artifact"]["id"] == "artifact-1"
    assert body["artifact"]["filename"] == "notes.txt"
    assert body["artifact"]["content_type"] == "text/plain"
    assert body["artifact"]["source_type"] == "file"
    assert body["artifact"]["status"] == "completed"
    assert body["artifact"]["chunk_count"] == body["chunk_count"]

    assert len(body["chunks"]) == body["chunk_count"]
    assert body["chunks"][0]["artifact_id"] == "artifact-1"
    assert body["chunks"][0]["filename"] == "notes.txt"
    assert body["chunks"][0]["vector_store_id"] == body["chunks"][0]["id"]
    assert "embedding" not in body["chunks"][0]

    artifact = metadata_store.get_artifact("artifact-1")
    assert artifact is not None
    assert artifact.status == "completed"
    assert artifact.chunk_count == body["chunk_count"]

    stored_chunks = metadata_store.get_chunks("artifact-1")
    assert len(stored_chunks) == body["chunk_count"]

    assert len(vector_store.added_chunks) == body["chunk_count"]


def test_failed_embedding_marks_artifact_failed(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: FailingEmbedLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store

    client = TestClient(app)

    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-failed",
            "filename": "notes.txt",
            "content": "This should fail during embedding.",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 503

    artifact = metadata_store.get_artifact("artifact-failed")
    assert artifact is not None
    assert artifact.status == "failed"
    assert artifact.error_message is not None
    assert "embedding backend unavailable" in artifact.error_message

    assert vector_store.added_chunks == []

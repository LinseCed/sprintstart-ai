from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.errors import LLMUnavailableError
from tests.conftest import llm_required
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

FIXTURES_DIR = Path(__file__).parent.parent / "ingestion/fixtures"


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubVectorStore], Any, None]:
    llm = StubLLMClient()
    store = StubVectorStore()

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store

    yield TestClient(app), store

    app.dependency_overrides.clear()


@pytest.fixture
def real_client() -> Generator[TestClient, Any, None]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ingest_small_markdown_returns_single_chunk(
    client: tuple[TestClient, StubVectorStore],
) -> None:
    http_client, store = client
    content = (FIXTURES_DIR / "markdown_small_sample.md").read_text()

    response = http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "small-doc",
            "filename": "markdown_small_sample.md",
            "content": content,
        },
    )

    assert response.status_code == 200
    assert response.json()["artifact_id"] == "small-doc"
    assert response.json()["chunk_count"] == 1
    assert len(store.chunks) == 1


def test_ingest_large_markdown_returns_multiple_chunks(
    client: tuple[TestClient, StubVectorStore],
) -> None:
    http_client, store = client
    content = (FIXTURES_DIR / "markdown_large_sample.md").read_text()

    response = http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "large-doc",
            "filename": "markdown_large_sample.md",
            "content": content,
        },
    )

    assert response.status_code == 200
    assert response.json()["artifact_id"] == "large-doc"
    assert response.json()["chunk_count"] > 1
    assert len(store.chunks) == response.json()["chunk_count"]


def test_reingest_replaces_existing_chunks(
    client: tuple[TestClient, StubVectorStore],
) -> None:
    http_client, store = client
    content = (FIXTURES_DIR / "markdown_small_sample.md").read_text()

    http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "doc-1",
            "filename": "markdown_small_sample.md",
            "content": content,
        },
    )
    response = http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "doc-1",
            "filename": "markdown_small_sample.md",
            "content": content,
        },
    )

    assert response.status_code == 200
    assert len([c for c in store.chunks if c.artifact_id == "doc-1"]) == 1


def test_ingest_llm_unavailable_returns_503(
    client: tuple[TestClient, StubVectorStore],
) -> None:
    http_client, _ = client

    def unavailable_llm() -> StubLLMClient:
        class FailingLLM(StubLLMClient):
            def embed(self, text: str) -> list[float]:
                raise LLMUnavailableError("http://localhost:11434")

        return FailingLLM()

    app.dependency_overrides[get_llm] = unavailable_llm
    content = (FIXTURES_DIR / "markdown_small_sample.md").read_text()

    response = http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "doc-1",
            "filename": "markdown_small_sample.md",
            "content": content,
        },
    )

    assert response.status_code == 503


@pytest.mark.integration
@llm_required
def test_ingest_small_markdown_with_real_llm(real_client: TestClient) -> None:
    content = (FIXTURES_DIR / "markdown_small_sample.md").read_text()

    response = real_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "small-doc",
            "filename": "markdown_small_sample.md",
            "content": content,
        },
    )

    assert response.status_code == 200
    assert response.json()["artifact_id"] == "small-doc"
    assert response.json()["chunk_count"] == 1

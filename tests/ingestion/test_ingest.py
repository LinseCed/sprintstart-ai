from pathlib import Path
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from tests.conftest import llm_required
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

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

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_ingest_small_markdown_returns_single_chunk(
    client: tuple[TestClient, StubVectorStore],
) -> None:
    http_client, store = client
    content = (FIXTURES_DIR / "markdown_small_sample.md").read_text()

    response = http_client.post(
        "/api/v1/ingest",
        json={"artifact_id": "small-doc", "filename": "markdown_small_sample.md", "content": content},
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
        json={"artifact_id": "large-doc", "filename": "markdown_large_sample.md", "content": content},
    )

    assert response.status_code == 200
    assert response.json()["artifact_id"] == "large-doc"
    assert response.json()["chunk_count"] > 1
    assert len(store.chunks) == response.json()["chunk_count"]

@llm_required
def test_ingest_small_markdown_with_real_ollama(real_client: TestClient) -> None:
    content = (FIXTURES_DIR / "markdown_small_sample.md").read_text()

    response = real_client.post(
      "/api/v1/ingest",
       json={"artifact_id": "0", "filename": "markdown_small_sample.md", "content": content},
    )

    assert response.status_code == 200
    assert response.json()["artifact_id"] == "0"
    assert response.json()["chunk_count"] == 1
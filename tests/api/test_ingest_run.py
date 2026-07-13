from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from ingestion.metadata_store import IngestionMetadataStore
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


@pytest.fixture
def vector_store() -> StubVectorStore:
    return StubVectorStore()


@pytest.fixture
def metadata_store(tmp_path: Path) -> Iterable[IngestionMetadataStore]:
    store = IngestionMetadataStore(path=str(tmp_path / "metadata.db"))
    yield store
    store.close()


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


Artifact = dict[str, object]


def _file_artifact(artifact_id: str = "uuid-1") -> Artifact:
    return {
        "artifactId": artifact_id,
        "sourceSystem": "GITHUB",
        "sourceId": "github:owner/repo:FILE:src/main/App.kt",
        "sourceUrl": "https://github.com/owner/repo/blob/main/src/main/App.kt",
        "artifactType": "FILE",
        "title": None,
        "bodyText": 'fun main() { println("hello") }',
        "mime": None,
        "language": "kotlin",
    }


def _issue_artifact(artifact_id: str = "uuid-2") -> Artifact:
    return {
        "artifactId": artifact_id,
        "sourceSystem": "GITHUB",
        "sourceId": "github:owner/repo:ISSUE:42",
        "sourceUrl": "https://github.com/owner/repo/issues/42",
        "artifactType": "ISSUE",
        "title": "Bug: login fails on mobile",
        "bodyText": "Steps to reproduce: open the app on iOS and tap login.",
        "mime": None,
        "language": None,
    }


def test_ingest_run_indexes_artifacts(
    client: TestClient,
    vector_store: StubVectorStore,
) -> None:
    response = client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [_file_artifact(), _issue_artifact()],
            "artifactsToDeindex": [],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["artifacts"]) == 2
    assert data["artifacts"][0]["artifact_id"] == "uuid-1"
    assert data["artifacts"][0]["chunk_count"] > 0
    assert data["artifacts"][1]["artifact_id"] == "uuid-2"
    assert data["artifacts"][1]["chunk_count"] > 0
    assert len(vector_store.chunks) == 2
    for chunk in vector_store.chunks:
        assert chunk.connector_id == "github"
        assert chunk.connector_source_id == "owner/repo"


def test_ingest_run_deindexes_before_indexing(
    client: TestClient,
    vector_store: StubVectorStore,
) -> None:
    # Seed an artifact that should be removed
    client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "old-artifact",
            "filename": "old.md",
            "content": "old content",
        },
    )
    assert vector_store.count_by_artifact("old-artifact") > 0

    response = client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [],
            "artifactsToDeindex": ["old-artifact"],
        },
    )

    assert response.status_code == 200
    assert vector_store.count_by_artifact("old-artifact") == 0


def test_ingest_run_empty_body_returns_empty_list(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingest/sync",
        json={"artifactsToIngest": [], "artifactsToDeindex": []},
    )

    assert response.status_code == 200
    assert response.json() == {"artifacts": []}


def test_ingest_run_artifact_with_no_content_returns_zero_chunks(
    client: TestClient,
) -> None:
    artifact: Artifact = _file_artifact()
    artifact["bodyText"] = None
    artifact["title"] = None

    response = client.post(
        "/api/v1/ingest/sync",
        json={"artifactsToIngest": [artifact], "artifactsToDeindex": []},
    )

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["chunk_count"] == 0


def test_ingest_run_filename_derived_from_source_id_for_files(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [_file_artifact("uuid-kt")],
            "artifactsToDeindex": [],
        },
    )

    record = metadata_store.get_artifact("uuid-kt")
    assert record is not None
    assert record.filename == "src/main/App.kt"


def test_ingest_run_filename_derived_for_issue(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [_issue_artifact("uuid-issue")],
            "artifactsToDeindex": [],
        },
    )

    record = metadata_store.get_artifact("uuid-issue")
    assert record is not None
    assert record.filename == "issue-42.md"

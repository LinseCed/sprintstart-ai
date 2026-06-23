import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_BASE = "/api/v1/onboarding/blueprints"
_SCOPE = "area:backend"


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    monkeypatch.setenv("BLUEPRINTS_PATH", str(tmp_path))
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "steps": [
                    {
                        "id": "deploy-runbook",
                        "title": "Read the deploy runbook",
                        "requirement": "required",
                        "chunk_ids": ["c1"],
                    }
                ]
            }
        )
    )
    llm.embedding = _EMBED
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="deploy.md",
                text="backend onboarding deploy runbook",
                embedding=_EMBED,
            )
        ]
    )

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), llm, store
    app.dependency_overrides.clear()


def _generate(http: TestClient) -> dict[str, Any]:
    response = http.post(f"{_BASE}/generate", json={"scopes": [_SCOPE]})
    assert response.status_code == 200, response.text
    return response.json()


def test_generate_then_list_drafts(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    body = _generate(http)
    assert body["outcomes"][0]["status"] == "created"

    drafts = http.get(f"{_BASE}/drafts").json()["items"]
    assert len(drafts) == 1
    assert drafts[0]["blueprint"]["scope"] == _SCOPE
    assert drafts[0]["blueprint"]["source"] == "generated"


def test_diff_endpoint(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client
    _generate(http)

    diff = http.get(f"{_BASE}/drafts/{_SCOPE}/diff")
    assert diff.status_code == 200
    body = diff.json()
    assert body["scope"] == _SCOPE
    assert any(c["change"] == "added" for c in body["changes"])


def test_approve_then_rollback(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, store = client
    _generate(http)

    approved = http.post(f"{_BASE}/drafts/{_SCOPE}/approve")
    assert approved.status_code == 200
    assert approved.json()["version"] == "1"

    # Re-generate after a corpus change to produce v2, then approve it.
    store.add(
        [
            Chunk(
                id="c2", artifact_id="a2", filename="x.md", text="new", embedding=_EMBED
            )
        ]
    )
    _generate(http)
    http.post(f"{_BASE}/drafts/{_SCOPE}/approve")

    versions = http.get(f"{_BASE}/{_SCOPE}/versions").json()["versions"]
    assert "1" in versions

    rolled = http.post(f"{_BASE}/{_SCOPE}/rollback", json={"version": "1"})
    assert rolled.status_code == 200
    assert rolled.json()["version"] == "1"


def test_diff_missing_draft_returns_404(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client
    response = http.get(f"{_BASE}/global/diff")
    assert response.status_code == 404

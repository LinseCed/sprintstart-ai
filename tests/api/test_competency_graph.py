import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_BASE = "/api/v1/onboarding/competency-graph"


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "competencies": [
                    {
                        "key": "kotlin",
                        "label": "Kotlin",
                        "kind": "SKILL",
                        "chunk_ids": ["c1"],
                    }
                ],
                "edges": [],
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
                filename="build.gradle.kts",
                text="Kotlin is the primary backend language",
                embedding=_EMBED,
            )
        ]
    )

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), llm, store
    app.dependency_overrides.clear()


def _propose(http: TestClient, **body: Any) -> dict[str, Any]:
    response = http.post(f"{_BASE}/propose", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_propose_returns_grounded_competencies(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    outcome = _propose(http)

    assert outcome["status"] == "proposed"
    assert outcome["competencies"][0]["key"] == "kotlin"
    assert outcome["provenance"]["corpus_fingerprint"] is not None


def test_propose_skips_active_competency(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    outcome = _propose(
        http,
        active_competencies=[{"key": "kotlin", "label": "Kotlin", "kind": "SKILL"}],
    )

    assert outcome["status"] == "skipped"
    assert outcome["competencies"] == []


def test_propose_unchanged_corpus_with_matching_fingerprint_is_a_noop(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    first = _propose(http)
    again = _propose(http, last_fingerprint=first["provenance"]["corpus_fingerprint"])

    assert again["status"] == "unchanged"

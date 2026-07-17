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
_BASE = "/api/v1/onboarding/lessons"


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "title": "Kotlin basics",
                "body": "Kotlin is the primary backend language.[c1]",
                "chunk_ids": ["c1"],
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


def test_synthesize_returns_grounded_lesson(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post(
        f"{_BASE}/synthesize",
        json={"competency_key": "kotlin", "competency_label": "Kotlin"},
    )

    assert response.status_code == 200, response.text
    outcome = response.json()
    assert outcome["status"] == "synthesized"
    assert outcome["lesson"]["competency_key"] == "kotlin"
    assert outcome["lesson"]["citations"][0]["chunk_id"] == "c1"


def test_synthesize_rejects_unknown_level(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post(
        f"{_BASE}/synthesize",
        json={
            "competency_key": "kotlin",
            "competency_label": "Kotlin",
            "level": "guru",
        },
    )

    assert response.status_code == 422

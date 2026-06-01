import json
from collections.abc import Generator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.conftest import llm_required
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def _parse_events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
    ]


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient()
    store = StubVectorStore()

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store

    yield TestClient(app), llm, store

    app.dependency_overrides.clear()


@pytest.fixture
def real_client() -> Generator[TestClient, Any, None]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_streams_tokens_and_done(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?"},
    )

    assert response.status_code == 200
    events = _parse_events(response.text)
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"


def test_chat_token_event_contains_llm_response(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, llm, _ = client
    llm.generate_response = "Missing designs and flaky CI."

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?"},
    )

    token_events = [e for e in _parse_events(response.text) if e["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["content"] == "Missing designs and flaky CI."


def test_chat_emits_citation_when_chunks_exist(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client
    # Use a non-zero embedding so cosine similarity is > 0
    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="retro.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                heading_path="Blockers",
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?", "min_score": 0.0},
    )

    citation_events = [
        e for e in _parse_events(response.text) if e["type"] == "citation"
    ]
    assert len(citation_events) == 1
    assert citation_events[0]["filename"] == "retro.md"
    assert citation_events[0]["chunk_id"] == "chunk-1"


def test_chat_with_history_succeeds(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/chat",
        json={
            "prompt": "Can you summarize that?",
            "context": [
                {"role": "user", "content": "What were the blockers?"},
                {"role": "assistant", "content": "Missing designs and flaky CI."},
            ],
        },
    )

    assert response.status_code == 200
    events = _parse_events(response.text)
    assert events[-1]["type"] == "done"


def test_chat_missing_question_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post("/api/v1/chat", json={})

    assert response.status_code == 422


def test_chat_llm_unavailable_emits_error_event(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    class StreamFailingLLM(StubLLMClient):
        def stream(self, messages: list[Message]) -> Iterator[str]:
            raise LLMUnavailableError("http://localhost:11434")

    app.dependency_overrides[get_llm] = lambda: StreamFailingLLM()

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?"},
    )

    # Error is emitted as an SSE event — HTTP status is still 200
    assert response.status_code == 200
    events = _parse_events(response.text)
    assert events[0]["type"] == "error"


@llm_required
def test_chat_with_real_llm(real_client: TestClient) -> None:
    response = real_client.post(
        "/api/v1/chat",
        json={"prompt": "Reply with one word: hello."},
    )

    assert response.status_code == 200
    events = _parse_events(response.text)
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"

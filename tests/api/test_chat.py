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
from tests.conftest import llm_required, parse_sse_events
from tests.stubs.llm import ScriptedLLMClient, StubLLMClient, Turn
from tests.stubs.store import StubVectorStore


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


def _parse_events(text: str) -> list[dict[str, object]]:
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
    ]


def test_chat_streams_tokens_and_done(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?"},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"


def test_chat_token_event_contains_llm_response(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, llm, store = client
    llm.generate_response = "Missing designs and flaky CI."

    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="retro.md",
                text="Missing designs and flaky CI.",
                embedding=embedding,
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?", "min_score": 0.0},
    )

    token_events = [e for e in parse_sse_events(response.text) if e["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["content"] == "Missing designs and flaky CI."


def test_chat_emits_citation_when_chunks_exist(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client
    # Use a non-zero embedding so cosine similarity is > 0
    embedding = [1.0] + [0.0] * 767
    script: list[Turn] = [
        [("synthesis", {"task": "blockers"})],
        [("retrieve", {"query": "blockers"})],
        [],
        [],
    ]
    app.dependency_overrides[get_llm] = lambda: ScriptedLLMClient(
        script, embedding=embedding
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="retro.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?"},
    )

    events = parse_sse_events(response.text)
    citation_events = [e for e in events if e["type"] == "citation"]
    assert len(citation_events) == 1
    assert citation_events[0]["filename"] == "retro.md"
    assert citation_events[0]["chunk_id"] == "chunk-1"

    tool_uses = [
        {"name": e["name"], "kind": e["kind"]}
        for e in events
        if e["type"] == "tool_use"
    ]
    assert tool_uses == [
        {"name": "synthesis", "kind": "agent"},
        {"name": "retrieve", "kind": "tool"},
        {"name": "retrieve", "kind": "tool"},
    ]
    assert events[-1] == {"type": "done"}


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
    events = parse_sse_events(response.text)
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
    http_client, _, store = client

    embedding = [1.0] + [0.0] * 767

    class StreamFailingLLM(StubLLMClient):
        def __init__(self) -> None:
            super().__init__(embedding=embedding)

        def stream(self, messages: list[Message]) -> Iterator[str]:
            raise LLMUnavailableError("http://localhost:11434")

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="retro.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
            )
        ]
    )

    app.dependency_overrides[get_llm] = lambda: StreamFailingLLM()

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?", "min_score": 0.0},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[0]["type"] == "error"


@pytest.mark.integration
@llm_required
def test_chat_with_real_llm(real_client: TestClient) -> None:
    response = real_client.post(
        "/api/v1/chat",
        json={"prompt": "Reply with one word: hello."},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"


def test_chat_with_filter_no_matching_chunks_returns_fallback(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client

    store.add(
        [
            Chunk(
                id="chunk-docs",
                artifact_id="artifact-docs",
                filename="doc.md",
                text="Docs text",
                embedding=[1.0] + [0.0] * 767,
                source_type="docs",
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "question": "What changed in code?",
            "min_score": 0.0,
            "filters": {"source_type": "code"},
        },
    )

    assert response.status_code == 200

    events = _parse_events(response.text)
    assert events[0]["type"] == "token"
    content = events[0]["content"]
    assert isinstance(content, str)
    assert "could not find any matching sources" in content
    assert events[-1]["type"] == "done"


def test_chat_with_source_filter_uses_matching_chunks(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client

    embedding = [1.0] + [0.0] * 767

    store.add(
        [
            Chunk(
                id="chunk-docs",
                artifact_id="artifact-docs",
                filename="doc.md",
                text="Docs text",
                embedding=embedding,
                source_type="docs",
            ),
            Chunk(
                id="chunk-code",
                artifact_id="artifact-code",
                filename="app.py",
                text="Code text",
                embedding=embedding,
                source_type="code",
                kind="code",
            ),
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "question": "What changed in code?",
            "min_score": 0.0,
            "filters": {"source_type": "code"},
        },
    )

    assert response.status_code == 200

    events = _parse_events(response.text)
    citation_events = [event for event in events if event["type"] == "citation"]

    assert len(citation_events) == 1
    assert citation_events[0]["chunk_id"] == "chunk-code"


def test_chat_without_chunks_returns_fallback_without_llm_hallucination(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/chat",
        json={"question": "What changed?"},
    )

    assert response.status_code == 200

    events = _parse_events(response.text)
    citation_events = [event for event in events if event["type"] == "citation"]

    assert events[0]["type"] == "token"
    content = events[0]["content"]
    assert isinstance(content, str)
    assert "could not find any matching sources" in content
    assert citation_events == []
    assert events[-1]["type"] == "done"

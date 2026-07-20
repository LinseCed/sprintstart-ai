from collections.abc import Generator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_source_state_store, get_store
from ingestion.source_state_store import SourceStateStore
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def _draft_markers(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The `tool_use` markers that flag the hand-off ("draft the question") mode."""
    return [
        e
        for e in events
        if e["type"] == "tool_use" and e.get("name") == "draft_question"
    ]


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient()
    store = StubVectorStore()

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store

    yield TestClient(app), llm, store

    app.dependency_overrides.clear()


def test_buddy_streams_tokens_and_done(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="onboarding.md",
                text="New hires get a laptop on day one.",
                embedding=embedding,
            )
        ]
    )

    response = http_client.post(
        "/api/v1/onboarding/buddy",
        json={"question": "How do I get set up?"},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"


def test_buddy_emits_citation_when_chunks_exist(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="onboarding.md",
                text="New hires get a laptop on day one.",
                embedding=embedding,
                start_line=3,
            )
        ]
    )

    response = http_client.post(
        "/api/v1/onboarding/buddy",
        json={"question": "How do I get set up?"},
    )

    events = parse_sse_events(response.text)
    citation_events = [e for e in events if e["type"] == "citation"]
    assert len(citation_events) == 1
    assert citation_events[0]["artifact_id"] == "doc-1"
    assert citation_events[0]["start_line"] == 3


def test_buddy_without_grounding_drafts_a_question_to_ask_a_human(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    # No indexed chunks -> the buddy must not guess. It hands off: a `draft_question`
    # marker so a client can tell the mode, the drafted text, then done. Nothing is
    # grounded, so no citations follow.
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/onboarding/buddy",
        json={"question": "What's the deploy process?"},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    types = [e["type"] for e in events]

    assert len(_draft_markers(events)) == 1
    assert "token" in types
    assert [e for e in events if e["type"] == "citation"] == []
    assert types[-1] == "done"


def test_buddy_grounded_answer_is_not_a_handoff(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    # A strong match is answered from the corpus (with a citation), never handed off.
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="onboarding.md",
                text="New hires get a laptop on day one.",
                embedding=embedding,
                start_line=1,
            )
        ]
    )

    response = http_client.post(
        "/api/v1/onboarding/buddy",
        json={"question": "How do I get set up?"},
    )

    events = parse_sse_events(response.text)
    assert _draft_markers(events) == []
    assert len([e for e in events if e["type"] == "citation"]) == 1
    assert events[-1]["type"] == "done"


def test_buddy_with_history_succeeds(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/onboarding/buddy",
        json={
            "question": "Can you say more about that?",
            "history": [
                {"role": "user", "content": "How do I get set up?"},
                {"role": "assistant", "content": "Start with the README."},
            ],
        },
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[-1]["type"] == "done"


def test_buddy_missing_question_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post("/api/v1/onboarding/buddy", json={})

    assert response.status_code == 422


def test_buddy_llm_unavailable_emits_error_event(
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
                filename="onboarding.md",
                text="New hires get a laptop on day one.",
                embedding=embedding,
            )
        ]
    )
    app.dependency_overrides[get_llm] = lambda: StreamFailingLLM()

    response = http_client.post(
        "/api/v1/onboarding/buddy",
        json={"question": "How do I get set up?"},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    # tool_use precedes the failure -- buddy always retrieves first, unlike
    # /chat's unfiltered ChatOrchestrator path which may fail before any
    # tool_use event.
    assert [e["type"] for e in events] == ["tool_use", "error"]


def test_buddy_applies_source_exclusions(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    """Same source-exclusion regression coverage as /chat's filtered path --
    a disabled connector/source must never surface via the buddy either."""
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)

    source_state = SourceStateStore(path=":memory:")
    source_state.set_connector_enabled("github", enabled=False)
    app.dependency_overrides[get_source_state_store] = lambda: source_state

    store.add(
        [
            Chunk(
                id="chunk-excluded",
                artifact_id="artifact-excluded",
                filename="excluded.md",
                text="New hires get a laptop on day one.",
                embedding=embedding,
                connector_id="github",
                connector_source_id="owner/repo",
            ),
            Chunk(
                id="chunk-included",
                artifact_id="artifact-included",
                filename="included.md",
                text="New hires get a laptop on day one.",
                embedding=embedding,
                connector_id="jira",
                connector_source_id="PROJ",
            ),
        ]
    )

    response = http_client.post(
        "/api/v1/onboarding/buddy",
        json={"question": "How do I get set up?"},
    )

    events = parse_sse_events(response.text)
    citation_events = [e for e in events if e["type"] == "citation"]
    assert [e["artifact_id"] for e in citation_events] == ["artifact-included"]

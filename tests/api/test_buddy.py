"""Route-level contract for the agentic buddy endpoint, including compaction."""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_source_state_store, get_store
from ingestion.source_state_store import SourceStateStore
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_URL = "/api/v1/onboarding/buddy/agent"


@pytest.fixture
def client() -> Generator[TestClient, Any, None]:
    llm = StubLLMClient(generate_response="condensed memory note")
    store = StubVectorStore()
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_source_state_store] = lambda: SourceStateStore(
        ":memory:"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_compaction_round_trips_through_the_endpoint(client: TestClient) -> None:
    response = client.post(
        _URL,
        json={
            "messages": [
                {"role": "user", "content": "m1"},
                {"role": "user", "content": "m2"},
                {"role": "user", "content": "m3"},
            ],
            "prior_summary": "old notes",
            "summarize_upto": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final"] is True
    assert body["updated_summary"] == "condensed memory note"
    # The folded turns are out of the returned window; the summary stands in, folded
    # into the system message the backend will carry back verbatim on a resume.
    assert body["messages"][0]["role"] == "system"
    assert "condensed memory note" in body["messages"][0]["content"]
    contents = [m["content"] for m in body["messages"]]
    assert "m1" not in contents
    assert "m3" in contents


def test_a_plain_turn_returns_no_updated_summary(client: TestClient) -> None:
    response = client.post(
        _URL,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final"] is True
    assert body["updated_summary"] is None
    assert body["messages"][0]["role"] == "system"


def test_open_stream_sends_the_greeting_before_the_memory_note() -> None:
    """The visible half arrives as SSE tokens.

    The private note only rides the terminal event.
    """
    llm = StubLLMClient(
        generate_response=(
            "Welcome back, Sam!\n<<<MEMORY>>>\nSam merged their first PR."
        )
    )
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/onboarding/buddy/open/stream",
                json={"memory": "old note", "recent": [], "state": "1 open PR"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "token"' in body
    assert "Welcome back, Sam!" in body
    # ⚠️ The note must never appear in a token event -- only in the terminal `done`.
    token_lines = [line for line in body.splitlines() if '"type": "token"' in line]
    assert all("Sam merged" not in line for line in token_lines)
    assert '"type": "done"' in body
    assert "Sam merged their first PR." in body

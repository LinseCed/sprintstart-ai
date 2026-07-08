from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


class CapturingLLM(StubLLMClient):
    def __init__(self) -> None:
        super().__init__(embedding=[1.0] + [0.0] * 767)
        self.messages: list[list[Message]] = []
        self.generate_response = (
            "## Key points\n"
            "- The artifact describes the onboarding flow.\n\n"
            "## Decisions\n"
            "- Keep the local-first setup.\n\n"
            "## What changed\n"
            "- The setup instructions were clarified."
        )

    def generate(self, messages: list[Message]) -> str:
        self.messages.append(messages)
        return self.generate_response


class FailingLLM(CapturingLLM):
    def generate(self, messages: list[Message]) -> str:
        raise LLMUnavailableError("local LLM unavailable")


@pytest.fixture
def client() -> Generator[tuple[TestClient, CapturingLLM, StubVectorStore], Any, None]:
    llm = CapturingLLM()
    store = StubVectorStore()

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store

    yield TestClient(app), llm, store

    app.dependency_overrides.clear()


def test_summary_returns_key_points_and_citation(
    client: tuple[TestClient, CapturingLLM, StubVectorStore],
) -> None:
    http_client, _, store = client

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="onboarding.md",
                text="The onboarding flow explains local setup and decisions.",
                embedding=[1.0] + [0.0] * 767,
                source_url="https://example.test/onboarding.md",
            )
        ]
    )

    response = http_client.post("/api/v1/artifacts/artifact-1/summary", json={})

    assert response.status_code == 200

    body = response.json()
    assert "Key points" in body["summary"]
    assert body["citations"] == [
        {
            "artifact_id": "artifact-1",
            "filename": "onboarding.md",
            "source_url": "https://example.test/onboarding.md",
        }
    ]


def test_summary_without_source_url_returns_backend_chunk_link(
    client: tuple[TestClient, CapturingLLM, StubVectorStore],
) -> None:
    http_client, _, store = client

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="notes.md",
                text="Important project notes.",
                embedding=[1.0] + [0.0] * 767,
            )
        ]
    )

    response = http_client.post("/api/v1/artifacts/artifact-1/summary", json={})

    assert response.status_code == 200
    body = response.json()

    assert body["citations"][0]["source_url"] == (
        "/api/v1/vector-db/artifacts/artifact-1/chunks"
    )


def test_summary_with_previous_artifact_mentions_change_context(
    client: tuple[TestClient, CapturingLLM, StubVectorStore],
) -> None:
    http_client, llm, store = client

    store.add(
        [
            Chunk(
                id="old-chunk",
                artifact_id="artifact-v1",
                filename="guide.md",
                text="Old setup used manual configuration.",
                embedding=[1.0] + [0.0] * 767,
            ),
            Chunk(
                id="new-chunk",
                artifact_id="artifact-v2",
                filename="guide.md",
                text="New setup uses automated configuration.",
                embedding=[1.0] + [0.0] * 767,
            ),
        ]
    )

    response = http_client.post(
        "/api/v1/artifacts/artifact-v2/summary",
        json={"previousArtifactId": "artifact-v1"},
    )

    assert response.status_code == 200

    prompt_text = "\n".join(
        message["content"] for message_list in llm.messages for message in message_list
    )

    assert "Use only the provided source excerpts" in prompt_text
    assert "Do not use external knowledge" in prompt_text
    assert "Previous version excerpts" in prompt_text
    assert "Old setup used manual configuration" in prompt_text
    assert "New setup uses automated configuration" in prompt_text

    body = response.json()
    assert len(body["citations"]) == 2


def test_summary_respects_max_chunks(
    client: tuple[TestClient, CapturingLLM, StubVectorStore],
) -> None:
    http_client, llm, store = client

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="notes.md",
                text="Included chunk text.",
                embedding=[1.0] + [0.0] * 767,
                position=0,
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-1",
                filename="notes.md",
                text="Excluded chunk text.",
                embedding=[1.0] + [0.0] * 767,
                position=1,
            ),
        ]
    )

    response = http_client.post(
        "/api/v1/artifacts/artifact-1/summary",
        json={"maxChunks": 1},
    )

    assert response.status_code == 200

    prompt_text = "\n".join(
        message["content"] for message_list in llm.messages for message in message_list
    )

    assert "Included chunk text." in prompt_text
    assert "Excluded chunk text." not in prompt_text


def test_summary_previous_artifact_missing_returns_404(
    client: tuple[TestClient, CapturingLLM, StubVectorStore],
) -> None:
    http_client, _, store = client

    store.add(
        [
            Chunk(
                id="new-chunk",
                artifact_id="artifact-v2",
                filename="guide.md",
                text="New setup uses automated configuration.",
                embedding=[1.0] + [0.0] * 767,
            )
        ]
    )

    response = http_client.post(
        "/api/v1/artifacts/artifact-v2/summary",
        json={"previousArtifactId": "missing"},
    )

    assert response.status_code == 404
    assert "Previous artifact" in response.json()["detail"]


def test_summary_unknown_artifact_returns_404(
    client: tuple[TestClient, CapturingLLM, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post("/api/v1/artifacts/missing/summary", json={})

    assert response.status_code == 404


def test_summary_llm_unavailable_returns_503(
    client: tuple[TestClient, CapturingLLM, StubVectorStore],
) -> None:
    http_client, _, store = client

    failing_llm = FailingLLM()
    app.dependency_overrides[get_llm] = lambda: failing_llm

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="notes.md",
                text="Important project notes.",
                embedding=[1.0] + [0.0] * 767,
            )
        ]
    )

    response = http_client.post("/api/v1/artifacts/artifact-1/summary", json={})

    assert response.status_code == 503
    assert "local LLM unavailable" in response.json()["detail"]


def test_long_artifact_is_batched(
    client: tuple[TestClient, CapturingLLM, StubVectorStore],
) -> None:
    http_client, llm, store = client

    store.add(
        [
            Chunk(
                id=f"chunk-{index}",
                artifact_id="artifact-long",
                filename="long.md",
                text="Long content. " * 300,
                embedding=[1.0] + [0.0] * 767,
                position=index,
            )
            for index in range(20)
        ]
    )

    response = http_client.post("/api/v1/artifacts/artifact-long/summary", json={})

    assert response.status_code == 200
    assert len(llm.messages) > 1

    prompt_text = "\n".join(
        message["content"] for message_list in llm.messages for message in message_list
    )

    assert "Summarize batch" in prompt_text
    assert "Use only the provided source excerpts" in prompt_text

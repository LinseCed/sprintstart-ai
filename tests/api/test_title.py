from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient], Any, None]:
    llm = StubLLMClient()

    app.dependency_overrides[get_llm] = lambda: llm

    yield TestClient(app), llm

    app.dependency_overrides.clear()


def test_title_generation(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client

    class TestLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            return "REST vs GraphQL"

    app.dependency_overrides[get_llm] = lambda: TestLLM()

    response = http_client.post(
        "/api/v1/generate-title",
        json={"prompt": "What are the differences between REST and GraphQL?"},
    )

    assert response.status_code == 200
    assert response.json() == {"title": "REST vs GraphQL"}


def test_title_respects_max_length(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client

    class LongTitleLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            return "THIS IS A VERY LONG TITLE THAT EXCEEDS LIMIT"

    app.dependency_overrides[get_llm] = lambda: LongTitleLLM()

    response = http_client.post(
        "/api/v1/generate-title", json={"prompt": "Generate title", "max_length": 10}
    )

    assert response.status_code == 200

    title = response.json()["title"]

    assert len(title) <= 10


def test_empty_prompt_returns_validation_error(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client

    response = http_client.post("/api/v1/generate-title", json={"prompt": "   "})

    assert response.status_code == 422


def test_llm_failure_returns_503(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client

    class FailingLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            raise LLMUnavailableError("http://localhost:11434")

    app.dependency_overrides[get_llm] = lambda: FailingLLM()

    response = http_client.post("/api/v1/generate-title", json={"prompt": "hello"})

    assert response.status_code == 503

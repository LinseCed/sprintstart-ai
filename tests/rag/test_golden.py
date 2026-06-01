"""End-to-end golden question / trick prompt test harness.

Ingests the demo corpus against a real Ollama instance, then:
- Asserts each golden question cites the expected source file.
- Asserts each trick prompt produces no citations (nothing relevant retrieved).

Marked pytest.mark.integration — skipped automatically when Ollama is not
reachable, so offline CI stays green.
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_store
from store.chroma_store import ChromaVectorStore
from tests.conftest import llm_required

CORPUS_DIR = Path(__file__).parent / "demo-corpus"
GOLDEN_QUESTIONS_FILE = Path(__file__).parent / "golden_questions.yaml"
TRICK_PROMPTS_FILE = Path(__file__).parent / "trick_prompts.yaml"

# Use a slightly relaxed threshold so small embedding drift doesn't flake.
_MIN_SCORE = 0.6


def _parse_events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
    ]


@pytest.fixture(scope="module")
def ingested_client() -> Generator[TestClient, Any, None]:
    """Ingest the full demo corpus once per module against a real LLM.

    Uses a fresh in-memory ChromaDB store so tests are isolated from any
    persistent data directory on the developer's machine.
    """
    store = ChromaVectorStore(collection_name="golden-test-chunks")
    app.dependency_overrides[get_store] = lambda: store

    client = TestClient(app)

    for doc_file in sorted(CORPUS_DIR.glob("*.md")):
        response = client.post(
            "/api/v1/ingest",
            json={
                "artifact_id": doc_file.stem,
                "filename": doc_file.name,
                "content": doc_file.read_text(),
            },
        )
        assert response.status_code == 200, (
            f"Corpus ingestion failed for {doc_file.name}: {response.text}"
        )

    yield client

    app.dependency_overrides.clear()


@pytest.mark.integration
@llm_required
def test_golden_questions(ingested_client: TestClient) -> None:
    """Every golden question must produce at least one citation from the
    expected file.
    """
    cases: list[dict[str, str]] = yaml.safe_load(GOLDEN_QUESTIONS_FILE.read_text())

    failures: list[str] = []
    for case in cases:
        question = case["question"]
        expected = case["expected_citation_filename"]

        response = ingested_client.post(
            "/api/v1/chat",
            json={"question": question, "min_score": _MIN_SCORE},
        )
        events = _parse_events(response.text)
        cited = {e["filename"] for e in events if e["type"] == "citation"}

        if expected not in cited:
            answer = "".join(e["content"] for e in events if e["type"] == "token")
            failures.append(
                f"\nQuestion : {question!r}"
                f"\nExpected : {expected}"
                f"\nCited    : {cited or '(none)'}"
                f"\nAnswer   : {answer!r}\n"
            )

    assert not failures, "Golden question assertion(s) failed:\n" + "".join(failures)


@pytest.mark.integration
@llm_required
def test_trick_prompts_return_no_citations(ingested_client: TestClient) -> None:
    """Trick prompts about absent content must produce no citations."""
    cases: list[dict[str, str]] = yaml.safe_load(TRICK_PROMPTS_FILE.read_text())

    failures: list[str] = []
    for case in cases:
        prompt = case["prompt"]

        response = ingested_client.post(
            "/api/v1/chat",
            json={"question": prompt, "min_score": _MIN_SCORE},
        )
        events = _parse_events(response.text)
        citation_events = [e for e in events if e["type"] == "citation"]

        if citation_events:
            cited = {e["filename"] for e in citation_events}
            answer = "".join(e["content"] for e in events if e["type"] == "token")
            failures.append(
                f"\nPrompt : {prompt!r}"
                f"\nExpected : no citations"
                f"\nGot      : {cited}"
                f"\nAnswer   : {answer!r}\n"
            )

    assert not failures, "Trick prompt(s) unexpectedly returned citations:\n" + "".join(
        failures
    )

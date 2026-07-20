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
_BASE = "/api/v1/onboarding/blueprints"
_SCOPE = "area:backend"
_CATALOG = [{"key": "deploy-runbook", "label": "Deploy the service", "kind": "SKILL"}]


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "competencies": [
                    {
                        "competency_key": "deploy-runbook",
                        "requirement": "required",
                        "rationale": "Every backend change ships through the runbook.",
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


def _generate(http: TestClient, **body: Any) -> dict[str, Any]:
    body.setdefault("scopes", [_SCOPE])
    body.setdefault("active_competencies", _CATALOG)
    response = http.post(f"{_BASE}/generate", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_generate_returns_a_competency_selection(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    outcome = _generate(http)["outcomes"][0]

    assert outcome["scope"] == _SCOPE
    assert outcome["status"] == "created"
    baseline = outcome["blueprint"]
    assert baseline["scope"] == _SCOPE
    assert baseline["source"] == "generated"
    assert baseline["version"] == "1"
    entry = baseline["competencies"][0]
    assert entry["competency_key"] == "deploy-runbook"
    assert entry["requirement"] == "required"
    assert entry["target_level"] is None


def test_generate_unchanged_corpus_is_a_noop(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    first = _generate(http)["outcomes"][0]["blueprint"]
    # The backend persists the result and passes it back as the active baseline.
    again = _generate(http, active=[first])["outcomes"][0]

    assert again["status"] == "unchanged"
    assert again["blueprint"] is None


def test_generate_bumps_version_against_active(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, store = client

    first = _generate(http)["outcomes"][0]["blueprint"]
    store.add(
        [
            Chunk(
                id="c2", artifact_id="a2", filename="x.md", text="new", embedding=_EMBED
            )
        ]
    )
    outcome = _generate(http, active=[first])["outcomes"][0]

    assert outcome["status"] == "updated"
    assert outcome["blueprint"]["version"] == "2"


def test_generate_without_a_catalog_is_skipped(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    """With no competency graph there is nothing to select, so nothing is proposed."""
    http, _, _ = client

    outcome = _generate(http, active_competencies=[])["outcomes"][0]

    assert outcome["status"] == "skipped"
    assert outcome["blueprint"] is None


def test_generate_discards_a_key_outside_the_catalog(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, llm, _ = client
    llm.generate_response = json.dumps(
        {
            "competencies": [
                {"competency_key": "made-up-key", "chunk_ids": ["c1"]},
                {"competency_key": "deploy-runbook", "chunk_ids": ["c1"]},
            ]
        }
    )

    outcome = _generate(http)["outcomes"][0]

    keys = [c["competency_key"] for c in outcome["blueprint"]["competencies"]]
    assert keys == ["deploy-runbook"]

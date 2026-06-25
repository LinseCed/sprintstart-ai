import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from onboarding.models import content_id
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

# Non-zero embedding so the stub store returns a perfect cosine match.
_EMBED = [1.0] + [0.0] * 767

_ACCOUNTS = "Set up your accounts and access"
_LOCAL_DB = "Set up your local database"


@pytest.fixture(autouse=True)
def _test_blueprints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    pool = [
        {"id": content_id(_ACCOUNTS), "title": _ACCOUNTS},
        {"id": content_id(_LOCAL_DB), "title": _LOCAL_DB},
    ]
    (tmp_path / "steps.yaml").write_text(yaml.safe_dump(pool))
    (tmp_path / "global.yaml").write_text(
        yaml.safe_dump(
            {
                "scope": "global",
                "version": "1",
                "source": "authored",
                "steps": [{"id": content_id(_ACCOUNTS), "requirement": "required"}],
            }
        )
    )
    (tmp_path / "area-backend.yaml").write_text(
        yaml.safe_dump(
            {
                "scope": "area:backend",
                "version": "1",
                "source": "authored",
                "steps": [{"id": content_id(_LOCAL_DB), "requirement": "required"}],
            }
        )
    )
    monkeypatch.setenv("BLUEPRINTS_PATH", str(tmp_path))


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient()
    store = StubVectorStore()

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store

    yield TestClient(app), llm, store

    app.dependency_overrides.clear()


def _post(http: TestClient, **body: Any) -> list[dict[str, Any]]:
    response = http.post("/api/v1/onboarding/path", json=body)
    assert response.status_code == 200
    return parse_sse_events(response.text)


def _path_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in events if e["type"] == "path")


def _all_step_ids(path: dict[str, Any]) -> list[str]:
    return [s["id"] for phase in path["phases"] for s in phase["steps"]]


def test_streams_stages_path_and_done(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    events = _post(http, working_area="backend", experience="junior")
    types = [e["type"] for e in events]

    assert "stage" in types
    assert types.count("path") == 1
    assert types[-1] == "done"


def test_required_steps_always_present(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    events = _post(http, working_area="backend", experience="junior")
    path = _path_event(events)["path"]
    ids = _all_step_ids(path)

    assert content_id("Set up your accounts and access") in ids
    assert content_id("Set up your local database") in ids


def test_unknown_working_area_falls_back_to_global_only(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    events = _post(http, working_area="unknown-area", experience="junior")
    path = _path_event(events)["path"]
    titles = [phase["title"] for phase in path["phases"]]

    assert titles == ["Getting started"]
    assert content_id("Set up your accounts and access") in _all_step_ids(path)


def test_unseen_experience_value_does_not_crash(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    events = _post(http, working_area="backend", experience="wizard")
    path = _path_event(events)["path"]

    assert content_id("Set up your accounts and access") in _all_step_ids(path)


def test_empty_corpus_produces_blueprint_only_path(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, store = client
    assert store.count() == 0

    events = _post(http, working_area="backend", experience="junior")
    path = _path_event(events)["path"]
    origins = {s["origin"] for phase in path["phases"] for s in phase["steps"]}

    assert origins == {"blueprint"}


def test_invalid_llm_output_falls_back_without_error(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, llm, store = client
    llm.embedding = _EMBED
    llm.generate_response = "this is not json at all"
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

    events = _post(http, working_area="backend", experience="junior")
    types = [e["type"] for e in events]
    path = _path_event(events)["path"]

    assert "error" not in types
    assert types[-1] == "done"
    origins = {s["origin"] for phase in path["phases"] for s in phase["steps"]}
    assert origins == {"blueprint"}


def test_grounded_llm_steps_are_added_and_cited(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, llm, store = client
    llm.embedding = _EMBED
    llm.generate_response = json.dumps(
        {
            "enriched": [
                {
                    "id": content_id("Set up your local database"),
                    "chunk_ids": ["c1"],
                }
            ],
            "added": [
                {
                    "title": "Read the deploy runbook",
                    "description": "How code ships to prod.",
                    "tags": ["deploy"],
                    "chunk_ids": ["c1"],
                }
            ],
        }
    )
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="deploy.md",
                text="backend onboarding deploy runbook local db",
                embedding=_EMBED,
            )
        ]
    )

    events = _post(http, working_area="backend", experience="junior")
    path = _path_event(events)["path"]
    quality = _path_event(events)["quality"]

    llm_steps = [
        s for phase in path["phases"] for s in phase["steps"] if s["origin"] == "llm"
    ]
    assert len(llm_steps) == 1
    assert llm_steps[0]["citations"][0]["filename"] == "deploy.md"
    assert quality["grounded_ratio"] == 1.0

    # Enriched blueprint step carries the citation too.
    db_step = next(
        s
        for phase in path["phases"]
        for s in phase["steps"]
        if s["id"] == content_id("Set up your local database")
    )
    assert db_step["citations"][0]["chunk_id"] == "c1"


def test_ungrounded_llm_step_is_dropped(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, llm, store = client
    llm.embedding = _EMBED
    # Added step references a chunk id that does not exist -> no citation -> dropped.
    llm.generate_response = json.dumps(
        {"enriched": [], "added": [{"title": "Ungrounded", "chunk_ids": ["missing"]}]}
    )
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="deploy.md",
                text="backend onboarding",
                embedding=_EMBED,
            )
        ]
    )

    events = _post(http, working_area="backend", experience="junior")
    path = _path_event(events)["path"]

    origins = {s["origin"] for phase in path["phases"] for s in phase["steps"]}
    assert origins == {"blueprint"}


def test_missing_request_field_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post("/api/v1/onboarding/path", json={"experience": "junior"})

    assert response.status_code == 422


# --- /path/yaml (synchronous YAML endpoint) ---


def test_yaml_endpoint_returns_valid_yaml(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post(
        "/api/v1/onboarding/path/yaml",
        json={"working_area": "backend", "experience": "junior"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-yaml"

    import yaml

    path = yaml.safe_load(response.text)
    assert path["working_area"] == "backend"
    assert path["experience"] == "junior"
    assert any(
        s["id"] == content_id("Set up your accounts and access")
        for phase in path["phases"]
        for s in phase["steps"]
    )


def test_yaml_endpoint_includes_quality_report(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post(
        "/api/v1/onboarding/path/yaml",
        json={"working_area": "backend", "experience": "junior"},
    )

    import yaml

    path = yaml.safe_load(response.text)
    assert "quality" in path
    assert "coverage" in path["quality"]
    assert "score" in path["quality"]


def test_yaml_endpoint_missing_field_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post("/api/v1/onboarding/path/yaml", json={"experience": "junior"})

    assert response.status_code == 422

import json
from pathlib import Path

import pytest
import yaml

from onboarding import drafts
from onboarding.generation import corpus_fingerprint, generate_blueprints
from onboarding.models import content_id
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

# Non-zero embedding so the stub store returns a perfect cosine match.
_EMBED = [1.0] + [0.0] * 767
_SCOPE = "area:backend"


def _llm(steps: list[dict[str, object]]) -> StubLLMClient:
    llm = StubLLMClient(generate_response=json.dumps({"steps": steps}))
    llm.embedding = _EMBED
    return llm


def _store(*texts: str) -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id="a1",
                filename=f"doc{i}.md",
                text=text,
                embedding=_EMBED,
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store


@pytest.fixture(autouse=True)
def tmp_blueprints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BLUEPRINTS_PATH", str(tmp_path))
    return tmp_path


def _seed_active(
    tmp_path: Path, *, skeleton: dict[str, object], pool: list[dict[str, object]]
) -> None:
    stem = "global" if skeleton["scope"] == "global" else "area-backend"
    (tmp_path / f"{stem}.yaml").write_text(yaml.safe_dump(skeleton), encoding="utf-8")
    (tmp_path / "steps.yaml").write_text(yaml.safe_dump(pool), encoding="utf-8")


def test_first_time_generation_drafts_grounded_steps() -> None:
    store = _store("backend onboarding deploy runbook local db setup")
    llm = _llm(
        [
            {
                "id": "deploy-runbook",
                "title": "Read the deploy runbook",
                "requirement": "required",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE])

    assert [(o.scope, o.status) for o in outcomes] == [(_SCOPE, "created")]
    draft = drafts.get_draft(_SCOPE)
    assert draft is not None
    assert draft.source == "generated"
    assert draft.version == "1"
    assert [s.id for s in draft.steps] == [content_id("Read the deploy runbook")]
    assert draft.steps[0].citations[0].chunk_id == "c1"
    assert draft.provenance is not None
    assert draft.provenance.corpus_fingerprint == corpus_fingerprint(store)


def test_unchanged_corpus_is_a_noop() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [{"id": "x", "title": "X", "requirement": "required", "chunk_ids": ["c1"]}]
    )

    generate_blueprints(llm, store, scopes=[_SCOPE])
    again = generate_blueprints(llm, store, scopes=[_SCOPE])

    assert again[0].status == "unchanged"


def test_corpus_change_updates_active_with_new_version() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [{"id": "x", "title": "X", "requirement": "required", "chunk_ids": ["c1"]}]
    )

    generate_blueprints(llm, store, scopes=[_SCOPE])
    drafts.approve_draft(_SCOPE)  # promote v1 to active

    store.add(
        [
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="d2.md",
                text="new",
                embedding=_EMBED,
            )
        ]
    )
    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE])

    assert outcomes[0].status == "updated"
    assert outcomes[0].draft_version == "2"


def test_invariant_removal_is_blocked_and_reinjected(tmp_path: Path) -> None:
    sec = "Security policy"
    _seed_active(
        tmp_path,
        skeleton={
            "scope": _SCOPE,
            "version": "1",
            "source": "authored",
            "steps": [{"id": content_id(sec), "requirement": "required"}],
        },
        pool=[{"id": content_id(sec), "title": sec}],
    )
    store = _store("backend onboarding deploy runbook")
    # The draft omits the required "security" step entirely.
    llm = _llm([{"title": "Deploy", "requirement": "recommended", "chunk_ids": ["c1"]}])

    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE])

    assert outcomes[0].status == "escalated"
    draft = drafts.get_draft(_SCOPE)
    assert draft is not None
    sec_id = content_id(sec)
    ids = {s.id for s in draft.steps}
    assert sec_id in ids  # protected step re-injected, never silently dropped
    assert draft.provenance is not None
    assert any(sec_id in note for note in draft.provenance.notes)


def test_invariant_flag_protects_recommended_step(tmp_path: Path) -> None:
    ethics = "Ethics training"
    _seed_active(
        tmp_path,
        skeleton={
            "scope": _SCOPE,
            "version": "1",
            "source": "authored",
            "steps": [
                {
                    "id": content_id(ethics),
                    "requirement": "recommended",
                    "invariant": True,
                }
            ],
        },
        pool=[{"id": content_id(ethics), "title": ethics}],
    )
    store = _store("backend onboarding")
    llm = _llm([{"title": "Deploy", "chunk_ids": ["c1"]}])

    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE])

    assert outcomes[0].status == "escalated"
    draft = drafts.get_draft(_SCOPE)
    assert draft is not None
    assert content_id(ethics) in {s.id for s in draft.steps}


def test_ungrounded_steps_are_dropped() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [
            {"id": "grounded", "title": "Grounded", "chunk_ids": ["c1"]},
            {"id": "ungrounded", "title": "Ungrounded", "chunk_ids": ["missing"]},
        ]
    )

    generate_blueprints(llm, store, scopes=[_SCOPE])

    draft = drafts.get_draft(_SCOPE)
    assert draft is not None
    assert [s.id for s in draft.steps] == [content_id("Grounded")]


def test_identical_title_proposals_dedup() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [
            {"title": "Read the deploy runbook", "chunk_ids": ["c1"]},
            {"title": "Read the deploy runbook", "chunk_ids": ["c1"]},
        ]
    )

    generate_blueprints(llm, store, scopes=[_SCOPE])

    draft = drafts.get_draft(_SCOPE)
    assert draft is not None
    # Same title -> same content id -> collapsed to a single step.
    assert [s.id for s in draft.steps] == [content_id("Read the deploy runbook")]


def test_empty_corpus_is_skipped() -> None:
    outcomes = generate_blueprints(_llm([]), StubVectorStore(), scopes=[_SCOPE])
    assert outcomes[0].status == "skipped"

import json

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from onboarding.generation import corpus_fingerprint
from onboarding.starter_work import generate_starter_work_pool
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767


def _metadata_store() -> IngestionMetadataStore:
    return IngestionMetadataStore(path=":memory:")


def _issue_artifact(**overrides: object) -> ArtifactRecord:
    defaults: dict[str, object] = dict(
        id="a1",
        filename="issue-1.md",
        content_type="text/plain",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        artifact_type="ISSUE",
        state="OPEN",
        source_id="github:org/repo:ISSUE:1",
        source_url="https://github.com/org/repo/issues/1",
        labels=["good first issue"],
    )
    defaults.update(overrides)
    return ArtifactRecord(**defaults)  # type: ignore[arg-type]


def _add_issue_chunk(
    store: StubVectorStore, artifact_id: str, title: str, body: str
) -> None:
    store.add(
        [
            Chunk(
                id=f"chunk-{artifact_id}",
                artifact_id=artifact_id,
                filename=f"{artifact_id}.md",
                text=f"# {title}\n\n{body}",
                embedding=_EMBED,
            )
        ]
    )


def _llm(tasks: list[dict[str, object]]) -> StubLLMClient:
    return StubLLMClient(generate_response=json.dumps({"tasks": tasks}))


def test_mines_safely_scoped_task_from_open_issue() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(
        store, "a1", "Fix typo in README", "The install section has a typo."
    )
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "Fix a typo in the README install section.",
                "competency_keys": ["docs"],
                "rationale": "Single-file text fix.",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "proposed"
    assert len(outcome.tasks) == 1
    task = outcome.tasks[0]
    assert task.source_id == "github:org/repo:ISSUE:1"
    assert task.title == "Fix typo in README"
    assert task.competency_keys == ["docs"]
    assert task.citations[0].source_url == "https://github.com/org/repo/issues/1"
    assert outcome.provenance is not None
    assert outcome.provenance.corpus_fingerprint == corpus_fingerprint(store)


def test_closed_issue_is_never_proposed() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact(state="CLOSED"))
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "x",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"
    assert outcome.tasks == []


def test_already_pooled_issue_is_not_reproposed() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "x",
            }
        ]
    )

    outcome = generate_starter_work_pool(
        llm, store, metadata_store, active_source_ids=["github:org/repo:ISSUE:1"]
    )

    assert outcome.status == "skipped"
    assert outcome.tasks == []


def test_competency_key_outside_known_set_is_dropped() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "x",
                "competency_keys": ["docs", "invented-key"],
            }
        ]
    )

    outcome = generate_starter_work_pool(
        llm, store, metadata_store, active_competency_keys=["docs"]
    )

    assert outcome.tasks[0].competency_keys == ["docs"]


def test_unchanged_corpus_with_matching_fingerprint_is_a_noop() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm([])

    outcome = generate_starter_work_pool(
        llm, store, metadata_store, last_fingerprint=corpus_fingerprint(store)
    )

    assert outcome.status == "unchanged"
    assert outcome.tasks == []


def test_empty_corpus_is_skipped() -> None:
    metadata_store = _metadata_store()
    store = StubVectorStore()
    llm = _llm([])

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"


def test_not_safely_scoped_is_dropped() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(
        store, "a1", "Rewrite the whole auth system", "big, vague, cross-cutting task"
    )
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": False,
                "rationale": "too large",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"
    assert outcome.tasks == []


def test_invalid_llm_json_is_skipped_not_raised() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = StubLLMClient(generate_response="not json at all")

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"


def test_issue_without_indexed_chunks_is_excluded() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(
        _issue_artifact(id="a1", source_id="github:org/repo:ISSUE:1")
    )
    metadata_store.save_artifact(
        _issue_artifact(id="a2", source_id="github:org/repo:ISSUE:2")
    )
    store = StubVectorStore()
    # Only a2 has been embedded; a1's issue text isn't in the vector store yet.
    _add_issue_chunk(store, "a2", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:2",
                "safely_scoped": True,
                "summary": "x",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert [t.source_id for t in outcome.tasks] == ["github:org/repo:ISSUE:2"]

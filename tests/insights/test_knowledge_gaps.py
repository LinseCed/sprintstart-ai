import json
from datetime import UTC, datetime, timedelta

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from insights.knowledge_gaps import (
    EXPECTED_TYPES,
    _component_of,
    _heuristic_present,
    _is_stale,
    _severity,
    detect_knowledge_gaps,
)
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_NOW = datetime.now(UTC).isoformat()


def _artifact(
    artifact_id: str,
    filename: str,
    *,
    source_id: str | None = None,
    updated_at: str = _NOW,
) -> ArtifactRecord:
    return ArtifactRecord(
        id=artifact_id,
        filename=filename,
        content_type="text/markdown",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at=_NOW,
        updated_at=updated_at,
        source_id=source_id,
    )


def _present_llm(present: list[str]) -> StubLLMClient:
    return StubLLMClient(generate_response=json.dumps({"present": present}))


# ── component derivation ─────────────────────────────────────────────────────


def test_component_of_extracts_owner_repo() -> None:
    record = _artifact("a1", "App.kt", source_id="github:acme/auth-service:FILE:App.kt")
    assert _component_of(record) == "acme/auth-service"


def test_component_of_returns_none_without_derivable_component() -> None:
    assert _component_of(_artifact("a1", "notes.md", source_id=None)) is None
    assert _component_of(_artifact("a2", "notes.md", source_id="freeform-id")) is None


# ── heuristic fallback classification ────────────────────────────────────────


def test_heuristic_present_maps_filenames_to_categories() -> None:
    records = [
        _artifact("a1", "README.md"),
        _artifact("a2", "runbook-deploy.md"),
        _artifact("a3", "openapi.yaml"),
    ]
    assert _heuristic_present(records) == {"readme", "runbook", "api"}


# ── severity heuristic ───────────────────────────────────────────────────────


def test_severity_high_when_critical_missing_and_many_gaps() -> None:
    missing = list(EXPECTED_TYPES)  # includes readme + setup (critical)
    assert _severity(missing, _NOW) == "high"


def test_severity_medium_for_single_critical_gap() -> None:
    # one missing critical category -> 1 + 2 = 3 -> medium
    assert _severity(["readme"], _NOW) == "medium"


def test_severity_low_for_single_noncritical_gap() -> None:
    assert _severity(["api"], _NOW) == "low"


def test_staleness_bumps_severity() -> None:
    stale = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    assert _is_stale(stale) is True
    # single non-critical gap (score 1) + stale (+1) -> 2 -> medium
    assert _severity(["api"], stale) == "medium"


# ── end-to-end detection ─────────────────────────────────────────────────────


def _store() -> IngestionMetadataStore:
    return IngestionMetadataStore(":memory:")


def test_detect_reports_missing_types_for_component() -> None:
    metadata_store = _store()
    metadata_store.save_completed_artifact(
        _artifact("a1", "README.md", source_id="github:acme/auth:FILE:README.md")
    )
    metadata_store.save_completed_artifact(
        _artifact("a2", "setup.md", source_id="github:acme/auth:FILE:setup.md")
    )

    gaps = detect_knowledge_gaps(
        _present_llm(["readme", "setup"]), StubVectorStore(), metadata_store
    )

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.component == "acme/auth"
    assert gap.present_types == ["readme", "setup"]
    assert gap.missing_types == ["architecture", "adr", "api", "runbook"]
    assert gap.severity == "high"


def test_detect_skips_fully_covered_component() -> None:
    metadata_store = _store()
    metadata_store.save_completed_artifact(
        _artifact("a1", "docs.md", source_id="github:acme/auth:FILE:docs.md")
    )

    gaps = detect_knowledge_gaps(
        _present_llm(list(EXPECTED_TYPES)), StubVectorStore(), metadata_store
    )

    assert gaps == []


def test_detect_skips_artifacts_without_component() -> None:
    metadata_store = _store()
    metadata_store.save_completed_artifact(_artifact("a1", "notes.md", source_id=None))

    gaps = detect_knowledge_gaps(
        _present_llm([]), StubVectorStore(), metadata_store
    )

    assert gaps == []


def test_detect_falls_back_to_heuristic_on_bad_llm_output() -> None:
    metadata_store = _store()
    metadata_store.save_completed_artifact(
        _artifact("a1", "README.md", source_id="github:acme/auth:FILE:README.md")
    )
    metadata_store.save_completed_artifact(
        _artifact("a2", "runbook.md", source_id="github:acme/auth:FILE:runbook.md")
    )

    gaps = detect_knowledge_gaps(
        StubLLMClient(generate_response="not json"), StubVectorStore(), metadata_store
    )

    assert len(gaps) == 1
    # heuristic picks readme + runbook from the filenames
    assert set(gaps[0].present_types) == {"readme", "runbook"}
    assert "setup" in gaps[0].missing_types

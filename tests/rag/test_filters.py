from ingestion.mapper import to_chunk
from ingestion.models import ParsedChunk
from rag.filters import matches_retrieval_filters
from rag.types import Chunk, RetrievalFilters


def test_source_system_filter_matches_allowed_system() -> None:
    chunk = Chunk(
        id="chunk-1",
        artifact_id="artifact-1",
        filename="app.py",
        text="Code",
        embedding=[1.0, 0.0],
        kind="code",
        source_system="GITHUB",
    )

    assert matches_retrieval_filters(
        chunk,
        RetrievalFilters(source_systems=["GITHUB", "JIRA"]),
    )
    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(source_systems=["UPLOAD"]),
    )


def test_source_timestamp_preferred_over_indexed_at() -> None:
    chunk = to_chunk(
        ParsedChunk(
            content="Old issue indexed today",
            kind="text",
            metadata={
                "filename": "issue-1.md",
                "indexed_at": "2026-06-01T00:00:00Z",
            },
        ),
        artifact_id="issue-1",
        embedding=[1.0, 0.0],
        artifact_type="ISSUE",
        source_updated_at="2025-01-01T00:00:00Z",
        source_system="JIRA",
    )

    assert chunk.created_at == "2025-01-01T00:00:00Z"
    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(time_from="2026-01-01T00:00:00Z"),
    )


def test_source_system_and_time_filters_are_combined_with_and() -> None:
    chunk = Chunk(
        id="chunk-1",
        artifact_id="artifact-1",
        filename="app.py",
        text="Code",
        embedding=[1.0, 0.0],
        kind="code",
        source_system="GITHUB",
        created_at="2026-03-01T00:00:00Z",
    )

    assert matches_retrieval_filters(
        chunk,
        RetrievalFilters(
            source_systems=["GITHUB"],
            time_from="2026-01-01T00:00:00Z",
            time_to="2026-07-01T00:00:00Z",
        ),
    )

    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(
            source_systems=["JIRA"],
            time_from="2026-01-01T00:00:00Z",
            time_to="2026-07-01T00:00:00Z",
        ),
    )

    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(
            source_systems=["GITHUB"],
            time_from="2026-04-01T00:00:00Z",
            time_to="2026-07-01T00:00:00Z",
        ),
    )

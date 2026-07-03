from datetime import UTC, datetime, timedelta

from ingestion.mapper import to_chunk
from ingestion.models import ParsedChunk
from rag.filters import matches_retrieval_filters, source_type_for_chunk
from rag.types import Chunk, RetrievalFilters


def test_pull_request_artifact_is_classified_as_ticket() -> None:
    chunk = to_chunk(
        ParsedChunk(
            content="Fixes auth bug",
            kind="text",
            metadata={"filename": "pull-request-123.md"},
        ),
        artifact_id="pr-123",
        embedding=[1.0, 0.0],
        artifact_type="PULL_REQUEST",
    )

    assert chunk.source_type == "tickets"
    assert source_type_for_chunk(chunk) == "tickets"


def test_source_timestamp_preferred_over_indexed_at() -> None:
    old_source_date = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    recent_index_date = datetime.now(UTC).isoformat()

    chunk = to_chunk(
        ParsedChunk(
            content="Old issue indexed today",
            kind="text",
            metadata={
                "filename": "issue-1.md",
                "indexed_at": recent_index_date,
            },
        ),
        artifact_id="issue-1",
        embedding=[1.0, 0.0],
        artifact_type="ISSUE",
        source_updated_at=old_source_date,
    )

    assert chunk.created_at == old_source_date
    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(time_range="last_6_months"),
    )


def test_source_type_and_time_range_filters_are_combined_with_and() -> None:
    recent_date = datetime.now(UTC).isoformat()

    chunk = Chunk(
        id="chunk-1",
        artifact_id="artifact-1",
        filename="app.py",
        text="Code",
        embedding=[1.0, 0.0],
        kind="code",
        source_type="code",
        created_at=recent_date,
    )

    assert matches_retrieval_filters(
        chunk,
        RetrievalFilters(source_type="code", time_range="last_6_months"),
    )
    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(source_type="tickets", time_range="last_6_months"),
    )

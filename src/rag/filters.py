from datetime import UTC, datetime, timedelta
from typing import Any

from rag.types import Chunk, RetrievalFilters, ScoredChunk, SourceType

_LATEST_DAYS = 30
_LAST_6_MONTHS_DAYS = 183


def timestamp_from_iso(value: str | None) -> float:
    if not value:
        return 0.0

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def cutoff_timestamp_for(time_range: str) -> float:
    now = datetime.now(UTC)

    if time_range == "latest":
        return (now - timedelta(days=_LATEST_DAYS)).timestamp()

    if time_range == "last_6_months":
        return (now - timedelta(days=_LAST_6_MONTHS_DAYS)).timestamp()

    return 0.0


def source_type_for_chunk(chunk: Chunk | ScoredChunk) -> SourceType:
    if chunk.source_type is not None:
        return chunk.source_type

    artifact_type = (chunk.artifact_type or "").upper()

    if chunk.kind == "code" or chunk.language:
        return "code"

    if artifact_type in {"ISSUE", "TICKET", "PULL_REQUEST"}:
        return "tickets"

    filename = chunk.filename.lower()
    if "ticket" in filename or "issue" in filename:
        return "tickets"

    return "docs"


def matches_retrieval_filters(
    chunk: Chunk | ScoredChunk,
    filters: RetrievalFilters | None,
) -> bool:
    if filters is None:
        return True

    if filters.source_type is not None:
        if source_type_for_chunk(chunk) != filters.source_type:
            return False

    if filters.time_range is not None:
        created_at_ts = timestamp_from_iso(chunk.created_at)
        if created_at_ts < cutoff_timestamp_for(filters.time_range):
            return False

    return True


def where_filter_for_chroma(filters: RetrievalFilters | None) -> Any | None:
    if filters is None:
        return None

    conditions: list[dict[str, object]] = []

    if filters.source_type is not None:
        conditions.append({"source_type": {"$eq": filters.source_type}})

    if filters.time_range is not None:
        conditions.append(
            {
                "created_at_ts": {
                    "$gte": cutoff_timestamp_for(filters.time_range),
                }
            }
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}

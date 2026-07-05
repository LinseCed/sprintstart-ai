from datetime import datetime
from typing import Any

from rag.types import (
    Chunk,
    RetrievalFilters,
    ScoredChunk,
    SourceSystem,
    is_source_system,
)


def normalize_source_system(value: str | None) -> SourceSystem | None:
    if value is None:
        return None

    normalized = value.upper()

    if is_source_system(normalized):
        return normalized

    return None


def timestamp_from_iso(value: str | None) -> float:
    parsed = _parse_timestamp(value)
    return parsed or 0.0


def _parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def matches_retrieval_filters(
    chunk: Chunk | ScoredChunk,
    filters: RetrievalFilters | None,
) -> bool:
    if filters is None:
        return True

    if filters.source_systems:
        if chunk.source_system not in filters.source_systems:
            return False

    has_time_filter = filters.time_from is not None or filters.time_to is not None

    if has_time_filter:
        chunk_timestamp = _parse_timestamp(chunk.created_at)

        if chunk_timestamp is None:
            return False

        if filters.time_from is not None:
            time_from = _parse_timestamp(filters.time_from)
            if time_from is not None and chunk_timestamp < time_from:
                return False

        if filters.time_to is not None:
            time_to = _parse_timestamp(filters.time_to)
            if time_to is not None and chunk_timestamp > time_to:
                return False

    return True


def where_filter_for_chroma(filters: RetrievalFilters | None) -> Any | None:
    if filters is None:
        return None

    conditions: list[dict[str, object]] = []

    if filters.source_systems:
        conditions.append({"source_system": {"$in": filters.source_systems}})

    has_time_filter = filters.time_from is not None or filters.time_to is not None

    if has_time_filter:
        conditions.append({"created_at_ts": {"$gt": 0.0}})

    if filters.time_from is not None:
        conditions.append(
            {"created_at_ts": {"$gte": timestamp_from_iso(filters.time_from)}}
        )

    if filters.time_to is not None:
        conditions.append(
            {"created_at_ts": {"$lte": timestamp_from_iso(filters.time_to)}}
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}

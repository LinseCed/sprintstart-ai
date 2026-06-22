import base64
import binascii
import logging
import mimetypes
import os
from dataclasses import replace
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import IngestRequest, ValidationErrorResponse
from ingestion.mapper import to_chunk
from ingestion.metadata_store import (
    ArtifactChunkRecord,
    ArtifactRecord,
    IngestionMetadataStore,
)
from ingestion.models import ParsedChunk
from ingestion.parser import parse
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestArtifactResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    source_type: str
    size_bytes: int
    chunk_count: int
    status: str
    created_at: str
    updated_at: str
    error_message: str | None = None


class IngestChunkResponse(BaseModel):
    id: str
    artifact_id: str
    filename: str
    text: str
    chunk_index: int
    vector_store_id: str
    kind: str


class IngestResponse(BaseModel):
    artifact_id: str
    chunk_count: int
    artifact: IngestArtifactResponse
    chunks: list[IngestChunkResponse]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _content_type_for_filename(filename: str) -> str:
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


def _artifact_to_response(artifact: ArtifactRecord) -> IngestArtifactResponse:
    return IngestArtifactResponse(
        id=artifact.id,
        filename=artifact.filename,
        content_type=artifact.content_type,
        source_type=artifact.source_type,
        size_bytes=artifact.size_bytes,
        chunk_count=artifact.chunk_count,
        status=artifact.status,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
        error_message=artifact.error_message,
    )


def _chunk_to_record(
    chunk: Chunk,
    chunk_index: int,
    created_at: str,
) -> ArtifactChunkRecord:
    return ArtifactChunkRecord(
        id=chunk.id,
        artifact_id=chunk.artifact_id,
        filename=chunk.filename,
        text=chunk.text,
        chunk_index=chunk_index,
        vector_store_id=chunk.id,
        kind=chunk.kind,
        created_at=created_at,
    )


def _chunk_to_response(chunk: ArtifactChunkRecord) -> IngestChunkResponse:
    return IngestChunkResponse(
        id=chunk.id,
        artifact_id=chunk.artifact_id,
        filename=chunk.filename,
        text=chunk.text,
        chunk_index=chunk.chunk_index,
        vector_store_id=chunk.vector_store_id,
        kind=chunk.kind,
    )


def _response_from_records(
    artifact: ArtifactRecord,
    chunks: list[ArtifactChunkRecord],
) -> IngestResponse:
    return IngestResponse(
        artifact_id=artifact.id,
        chunk_count=artifact.chunk_count,
        artifact=_artifact_to_response(artifact),
        chunks=[_chunk_to_response(chunk) for chunk in chunks],
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a document",
    description=(
        "Parses, chunks, and embeds a document, then stores it in the vector store. "
        "Re-ingesting the same artifact_id replaces the existing chunks. "
        "Supported types: plain text, Markdown, JSON, YAML, TOML, and images "
        "(.png, .jpg, .jpeg, .gif, .webp, .bmp — send as base64-encoded content). "
        "Image files are captioned via the configured vision model; if no vision model "
        "is available the request succeeds with chunk_count=0."
    ),
    responses={
        422: {
            "model": ValidationErrorResponse,
            "content": {
                "application/json": {"example": {"detail": "'question' is required"}}
            },
        },
        503: {
            "model": ValidationErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "detail": "LLM backend unreachable at 'http://localhost:11434'"
                    }
                }
            },
        },
    },
)
def ingest(
    body: IngestRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    store: Annotated[VectorStore, Depends(get_store)],
    metadata_store: Annotated[
        IngestionMetadataStore,
        Depends(get_ingestion_metadata_store),
    ],
) -> IngestResponse:
    content_bytes = body.content.encode("utf-8")
    now = _utc_now()

    artifact = ArtifactRecord(
        id=body.artifact_id,
        filename=body.filename,
        content_type=_content_type_for_filename(body.filename),
        source_type="file",
        size_bytes=len(content_bytes),
        chunk_count=0,
        status="processing",
        created_at=now,
        updated_at=now,
    )
    metadata_store.save_artifact(artifact)

    max_length = int(os.getenv("INGEST_MAX_CONTENT_LENGTH", "500000"))
    if len(body.content) > max_length:
        detail = f"Content exceeds maximum length of {max_length} characters."
        metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
        raise HTTPException(status_code=413, detail=detail)

    try:
        parsed_chunks = parse(body.filename, content_bytes)
    except NotImplementedError as exc:
        suffix = (
            body.filename.rsplit(".", 1)[-1] if "." in body.filename else body.filename
        )
        detail = f"Parsing .{suffix} files is not yet supported."
        metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
        raise HTTPException(status_code=422, detail=detail) from exc

    if not parsed_chunks:
        detail = f"Unsupported file type: {body.filename}"
        metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
        raise HTTPException(status_code=422, detail=detail)

    enriched: list[ParsedChunk] = []
    for chunk in parsed_chunks:
        if chunk.kind == "image":
            try:
                image_bytes = base64.b64decode(chunk.content, validate=True)
            except (binascii.Error, ValueError) as exc:
                detail = f"Image content for {body.filename!r} is not valid base64."
                metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
                raise HTTPException(status_code=422, detail=detail) from exc

            try:
                caption = llm.caption_image(image_bytes)
                enriched.append(
                    ParsedChunk(content=caption, kind="image", metadata=chunk.metadata)
                )
            except LLMUnavailableError:
                logger.warning(
                    "Vision model unavailable — skipping image chunk in %s",
                    body.filename,
                )
        else:
            enriched.append(chunk)

    if not enriched:
        try:
            store.delete(body.artifact_id, exclude_ids=[])
        except Exception as exc:
            detail = f"Failed to delete existing vector chunks: {exc}"
            metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
            raise HTTPException(status_code=500, detail=detail) from exc

        completed_artifact = replace(
            artifact,
            chunk_count=0,
            status="completed",
            updated_at=_utc_now(),
        )
        metadata_store.save_completed_artifact(completed_artifact, [])
        return _response_from_records(completed_artifact, [])

    try:
        chunks = [
            to_chunk(chunk, body.artifact_id, llm.embed(chunk.content))
            for chunk in enriched
        ]
    except LLMUnavailableError as exc:
        detail = str(exc)
        metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
        raise HTTPException(status_code=503, detail=detail) from exc

    try:
        store.add(chunks)
        store.delete(body.artifact_id, exclude_ids=[chunk.id for chunk in chunks])
    except Exception as exc:
        detail = f"Failed to store chunks in vector database: {exc}"
        metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
        raise HTTPException(status_code=500, detail=detail) from exc

    completed_artifact = replace(
        artifact,
        chunk_count=len(chunks),
        status="completed",
        updated_at=_utc_now(),
    )
    chunk_records = [
        _chunk_to_record(
            chunk,
            chunk_index=index,
            created_at=completed_artifact.updated_at,
        )
        for index, chunk in enumerate(chunks)
    ]

    metadata_store.save_completed_artifact(completed_artifact, chunk_records)

    return _response_from_records(completed_artifact, chunk_records)

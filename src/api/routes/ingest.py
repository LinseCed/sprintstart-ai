import base64
import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_llm, get_store
from api.schemas import IngestRequest, IngestResponse, ValidationErrorResponse
from ingestion.mapper import to_chunk
from ingestion.models import ParsedChunk
from ingestion.parser import parse
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ingest",
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
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> IngestResponse:
    max_length = int(os.getenv("INGEST_MAX_CONTENT_LENGTH", "500000"))
    if len(body.content) > max_length:
        raise HTTPException(
            status_code=413,
            detail=f"Content exceeds maximum length of {max_length} characters.",
        )

    parsed_chunks = parse(body.filename, body.content.encode("utf-8"))

    if not parsed_chunks:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {body.filename}",
        )

    # Caption pass: replace image chunk content with LLM-generated caption.
    # On LLMUnavailableError the chunk is dropped and ingestion continues.
    enriched: list[ParsedChunk] = []
    for chunk in parsed_chunks:
        if chunk.kind == "image":
            try:
                image_bytes = base64.b64decode(chunk.content)
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail=f"Image content for {body.filename!r} is not valid base64.",
                )
            try:
                caption = llm.caption_image(image_bytes)
                enriched.append(ParsedChunk(content=caption, kind="image", metadata=chunk.metadata))
            except LLMUnavailableError:
                logger.warning("Vision model unavailable — skipping image chunk in %s", body.filename)
        else:
            enriched.append(chunk)

    if not enriched:
        store.delete(body.artifact_id, exclude_ids=[])
        return IngestResponse(artifact_id=body.artifact_id, chunk_count=0)

    # Embed pass: generate embeddings and store.
    try:
        chunks = [
            to_chunk(chunk, body.artifact_id, llm.embed(chunk.content))
            for chunk in enriched
        ]
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    store.add(chunks)
    store.delete(body.artifact_id, exclude_ids=[c.id for c in chunks])

    return IngestResponse(artifact_id=body.artifact_id, chunk_count=len(chunks))

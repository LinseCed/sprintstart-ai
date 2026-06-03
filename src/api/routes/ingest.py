import os

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_llm, get_store
from api.schemas import IngestRequest, IngestResponse, ValidationErrorResponse
from ingestion.mapper import to_chunk
from ingestion.parser import parse
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

router = APIRouter()


@router.post(
    "/ingest",
    summary="Ingest a document",
    description=(
        "Parses, chunks, and embeds a document, then stores it in the vector store. "
        "Re-ingesting the same artifact_id replaces the existing chunks."
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

    try:
        parsed_chunks = parse(body.filename, body.content.encode("utf-8"))
    except NotImplementedError:
        suffix = body.filename.rsplit(".", 1)[-1] if "." in body.filename else body.filename
        raise HTTPException(
            status_code=422,
            detail=f"Parsing .{suffix} files is not yet supported.",
        )

    if not parsed_chunks:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {body.filename}",
        )

    try:
        chunks = [
            to_chunk(chunk, body.artifact_id, llm.embed(chunk.content))
            for chunk in parsed_chunks
        ]
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    store.add(chunks)
    store.delete(body.artifact_id, exclude_ids=[c.id for c in chunks])

    return IngestResponse(artifact_id=body.artifact_id, chunk_count=len(chunks))

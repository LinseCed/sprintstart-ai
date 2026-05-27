from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_llm, get_store
from api.schemas import IngestRequest, IngestResponse
from ingestion.mapper import to_chunk
from ingestion.parser import parse
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

router = APIRouter()


@router.post("/ingest")
def ingest(
    body: IngestRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> IngestResponse:
    parsed_chunks = parse(body.filename, body.content.encode("utf-8"))

    if not parsed_chunks:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {body.filename}")

    try:
        chunks = [
            to_chunk(chunk, body.artifact_id, llm.embed(chunk.content))
            for chunk in parsed_chunks
        ]
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    store.delete(body.artifact_id)
    store.add(chunks)

    return IngestResponse(artifact_id=body.artifact_id, chunk_count=len(chunks))

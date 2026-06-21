from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_llm, get_store
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from rag.types import Chunk, ScoredChunk
from store.base import VectorStore

router = APIRouter(prefix="/vector-db", tags=["vector-db"])


class ChunkResponse(BaseModel):
    id: str
    artifact_id: str
    text: str
    heading_path: list[str] = Field(default_factory=list)


class ChunkListResponse(BaseModel):
    items: list[ChunkResponse]
    limit: int
    offset: int
    total: int


class VectorDbStatusResponse(BaseModel):
    backend: str
    chunk_count: int


class DeleteArtifactResponse(BaseModel):
    artifact_id: str
    deleted: bool
    deleted_count: int


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0)


class VectorSearchResponse(BaseModel):
    items: list[ChunkResponse]


def _scored_chunk_to_response(scored_chunk: ScoredChunk) -> ChunkResponse:
    return ChunkResponse(
        id=scored_chunk.id,
        artifact_id=scored_chunk.artifact_id,
        text=scored_chunk.text,
        heading_path=_normalize_heading_path(scored_chunk.heading_path),
    )


def _chunk_to_response(chunk: Chunk) -> ChunkResponse:
    return ChunkResponse(
        id=chunk.id,
        artifact_id=chunk.artifact_id,
        text=chunk.text,
        heading_path=_normalize_heading_path(chunk.heading_path),
    )


def _normalize_heading_path(heading_path: object) -> list[str]:
    if heading_path is None:
        return []

    if isinstance(heading_path, str):
        return [heading_path]

    if isinstance(heading_path, list):
        typed_heading_path = cast(list[object], heading_path)
        return [str(item) for item in typed_heading_path]

    return []


def _store_backend_name(store: VectorStore) -> str:
    name = type(store).__name__.lower()

    if "chroma" in name:
        return "chroma"

    return name


@router.get("/status", response_model=VectorDbStatusResponse)
def get_vector_db_status(
    store: Annotated[VectorStore, Depends(get_store)],
) -> VectorDbStatusResponse:
    return VectorDbStatusResponse(
        backend=_store_backend_name(store),
        chunk_count=store.count(),
    )


@router.get("/chunks", response_model=ChunkListResponse)
def list_chunks(
    store: Annotated[VectorStore, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkListResponse:
    chunks = store.all_chunks()
    paginated_chunks = chunks[offset : offset + limit]

    return ChunkListResponse(
        items=[_chunk_to_response(chunk) for chunk in paginated_chunks],
        limit=limit,
        offset=offset,
        total=len(chunks),
    )


@router.get("/artifacts/{artifact_id}/chunks", response_model=list[ChunkResponse])
def list_chunks_by_artifact(
    artifact_id: str,
    store: Annotated[VectorStore, Depends(get_store)],
) -> list[ChunkResponse]:
    chunks = [chunk for chunk in store.all_chunks() if chunk.artifact_id == artifact_id]

    return [_chunk_to_response(chunk) for chunk in chunks]


@router.delete(
    "/artifacts/{artifact_id}",
    response_model=DeleteArtifactResponse,
)
def delete_artifact_chunks(
    artifact_id: str,
    store: Annotated[VectorStore, Depends(get_store)],
) -> DeleteArtifactResponse:
    before_count = store.count()

    store.delete(artifact_id)

    after_count = store.count()
    deleted_count = max(before_count - after_count, 0)

    return DeleteArtifactResponse(
        artifact_id=artifact_id,
        deleted=deleted_count > 0,
        deleted_count=deleted_count,
    )


@router.post("/search", response_model=VectorSearchResponse)
def search_vector_db(
    request: VectorSearchRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> VectorSearchResponse:
    try:
        embedding = llm.embed(request.query)
        chunks = store.query(
            embedding=embedding,
            top_k=request.top_k,
            min_score=request.min_score,
        )

        return VectorSearchResponse(
            items=[_scored_chunk_to_response(chunk) for chunk in chunks]
        )

    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

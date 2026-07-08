from fastapi import APIRouter, Depends, HTTPException, status

from agents.artifact_summary import ArtifactSummaryAgent
from api.dependencies import get_llm, get_store
from api.schemas import (
    ArtifactSummaryCitation,
    ArtifactSummaryRequest,
    ArtifactSummaryResponse,
)
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from store.base import VectorStore

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.post(
    "/{artifact_id}/summary",
    response_model=ArtifactSummaryResponse,
    summary="Summarize an artifact",
)
def summarize_artifact(
    artifact_id: str,
    body: ArtifactSummaryRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> ArtifactSummaryResponse:
    chunks = _chunks_for_artifact(store, artifact_id, body.max_chunks)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id!r} was not found or has no chunks.",
        )

    previous_chunks: list[Chunk] | None = None

    if body.previous_artifact_id is not None:
        previous_chunks = _chunks_for_artifact(
            store,
            body.previous_artifact_id,
            body.max_chunks,
        )

        if not previous_chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Previous artifact {body.previous_artifact_id!r} "
                    "was not found or has no chunks."
                ),
            )

    try:
        result = ArtifactSummaryAgent(llm).summarize(
            artifact_id=artifact_id,
            chunks=chunks,
            previous_chunks=previous_chunks,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return ArtifactSummaryResponse(
        artifact_id=result.artifact_id,
        summary=result.summary,
        citations=[
            ArtifactSummaryCitation(
                artifact_id=citation.artifact_id,
                filename=citation.filename,
                source_url=citation.source_url,
            )
            for citation in result.citations
        ],
    )


def _chunks_for_artifact(
    store: VectorStore,
    artifact_id: str,
    max_chunks: int,
) -> list[Chunk]:
    return store.list_chunks_by_artifact(
        artifact_id=artifact_id,
        limit=max_chunks,
        offset=0,
    )

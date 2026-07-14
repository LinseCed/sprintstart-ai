import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from agents.artifact_summary import ArtifactSummaryAgent, SummaryStage
from api.dependencies import get_llm, get_store
from api.schemas import ArtifactSummaryRequest
from api.sse import sse_event
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.post(
    "/{artifact_id}/summary",
    summary="Summarize an artifact (streaming)",
    response_class=StreamingResponse,
    description=(
        "Streams an AI-generated summary of the artifact over Server-Sent Events.\n\n"
        "Event sequence:\n"
        "1. `stage` events while notes are gathered from the source (not streamed "
        "itself -- internal working notes, not the final summary)\n"
        "2. `token` events streaming the final summary text\n"
        "3. One `citation` event per source cited in the summary\n"
        "4. Exactly one `done` event\n\n"
        "On error, a single `error` event is emitted instead and the stream closes."
    ),
)
def summarize_artifact(
    artifact_id: str,
    body: ArtifactSummaryRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> StreamingResponse:
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

    def event_stream() -> Iterator[str]:
        try:
            gen = ArtifactSummaryAgent(llm).summarize_stream(
                artifact_id=artifact_id,
                chunks=chunks,
                previous_chunks=previous_chunks,
            )
            try:
                while True:
                    event = next(gen)
                    if isinstance(event, SummaryStage):
                        yield sse_event(
                            {
                                "type": "stage",
                                "name": event.name,
                                "detail": event.detail,
                            }
                        )
                    else:
                        yield sse_event({"type": "token", "content": event})
            except StopIteration as stop:
                result = stop.value

            for citation in result.citations:
                yield sse_event(
                    {
                        "type": "citation",
                        "artifact_id": citation.artifact_id,
                        "filename": citation.filename,
                        "source_url": citation.source_url,
                    }
                )

            yield sse_event({"type": "done"})

        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in artifact summary stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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

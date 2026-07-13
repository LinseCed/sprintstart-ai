import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agents.orchestrator import ChatOrchestrator
from api.dependencies import get_llm, get_store
from api.schemas import ChatRequest, ValidationErrorResponse
from api.sse import sse_event
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from rag.prompt import build_messages
from rag.retriever import retrieve
from rag.types import RetrievalFilters
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

_FILTERED_TOP_K = 5
_FILTERED_MIN_SCORE = 0.3

_NO_FILTERED_RESULTS_MESSAGE = (
    "I could not find any matching sources for the selected filters, "
    "so I cannot answer this reliably."
)


def _retrieval_filters_from_request(body: ChatRequest) -> RetrievalFilters | None:
    if body.filters is None:
        return None

    source_systems = body.filters.source_systems or None

    if (
        source_systems is None
        and body.filters.time_from is None
        and body.filters.time_to is None
    ):
        return None

    return RetrievalFilters(
        source_systems=source_systems,
        time_from=body.filters.time_from,
        time_to=body.filters.time_to,
    )


@router.post(
    "/chat",
    summary="Ask a question (streaming)",
    response_class=StreamingResponse,
    responses={422: {"model": ValidationErrorResponse}},
)
def chat(
    body: ChatRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        try:
            filters = _retrieval_filters_from_request(body)

            if filters is None:
                yield from ChatOrchestrator(llm, store).stream(
                    body.question,
                    body.history,
                )
                return

            yield sse_event({"type": "tool_use", "name": "retrieve", "kind": "tool"})

            chunks = retrieve(
                body.question,
                llm,
                store,
                top_k=_FILTERED_TOP_K,
                min_score=_FILTERED_MIN_SCORE,
                filters=filters,
            )

            if not chunks:
                yield sse_event(
                    {
                        "type": "token",
                        "content": _NO_FILTERED_RESULTS_MESSAGE,
                    }
                )
                yield sse_event({"type": "done"})
                return

            history: list[Message] = [
                Message(role=h.role, content=h.content) for h in body.history
            ]
            messages = build_messages(body.question, chunks, history)

            for token in llm.stream(messages):
                if token:
                    yield sse_event({"type": "token", "content": token})

            for citation in build_citations(chunks):
                yield sse_event(
                    {
                        "type": "citation",
                        "chunk_id": citation.chunk_id,
                        "filename": citation.filename,
                        "source_url": citation.source_url,
                    }
                )

            yield sse_event({"type": "done"})

        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in chat stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")

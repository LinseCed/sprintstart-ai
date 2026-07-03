import json
import logging
from collections.abc import Iterator, Mapping

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agents.orchestrator import ChatOrchestrator
from api.dependencies import get_llm, get_store
from api.schemas import ChatRequest, ValidationErrorResponse
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from rag.prompt import build_messages
from rag.retriever import retrieve
from rag.types import RetrievalFilters
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

_NO_FILTERED_RESULTS_MESSAGE = (
    "I could not find any matching sources for the selected filters, "
    "so I cannot answer this reliably."
)


def _sse_event(payload: Mapping[str, object | None]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _retrieval_filters_from_request(body: ChatRequest) -> RetrievalFilters | None:
    if body.filters is None:
        return None

    return RetrievalFilters(
        source_type=body.filters.source_type,
        time_range=body.filters.time_range,
    )


@router.post(
    "/chat",
    summary="Ask a question (streaming)",
    response_class=StreamingResponse,
    description=(
        "Retrieves relevant chunks from the vector store, then streams a "
        "generated answer token by token as Server-Sent Events.\n\n"
        "Event sequence:\n"
        "1. Zero or more `token` events (the answer, in order)\n"
        "2. Zero or more `citation` events (sources used)\n"
        "3. Exactly one `done` event\n\n"
        "On error, a single `error` event is emitted instead and the stream closes."
    ),
    responses={
        200: {
            "description": "SSE stream",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "Newline-delimited SSE stream. Each event is a JSON "
                            "object. See TokenEvent, CitationEvent, DoneEvent, "
                            "ErrorEvent schemas."
                        ),
                    },
                    "examples": {
                        "token": {
                            "summary": "Token event",
                            "value": (
                                'data: {"type": "token", "content": "The main"}\n\n'
                            ),
                        },
                        "citation": {
                            "summary": "Citation event",
                            "value": (
                                'data: {"type": "citation", '
                                '"chunk_id": "chunk-1", '
                                '"filename": "retro.md", '
                                '"section_path": "Retro > Blockers"}\n\n'
                            ),
                        },
                        "done": {
                            "summary": "Done event",
                            "value": 'data: {"type": "done"}\n\n',
                        },
                        "error": {
                            "summary": "Error event",
                            "value": (
                                'data: {"type": "error", '
                                '"message": "LLM backend unreachable"}\n\n'
                            ),
                        },
                    },
                }
            },
        },
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
def chat(
    body: ChatRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        try:
            filters = _retrieval_filters_from_request(body)

            chunks = retrieve(
                body.question,
                llm,
                store,
                body.top_k,
                body.min_score,
                filters=filters,
            )

            if not chunks:
                message = (
                    _NO_FILTERED_RESULTS_MESSAGE
                    if filters is not None
                    else "I could not find relevant sources, "
                    "so I cannot answer this reliably."
                )
                yield _sse_event({"type": "token", "content": message})
                yield _sse_event({"type": "done"})
                return

            if filters is None:
                yield from ChatOrchestrator(llm, store).stream(
                    body.question,
                    body.history,
                )
                return

            history: list[Message] = [
                Message(role=h.role, content=h.content) for h in body.history
            ]
            messages = build_messages(body.question, chunks, history)

            for token in llm.stream(messages):
                if token:
                    yield _sse_event({"type": "token", "content": token})

            for citation in build_citations(chunks):
                payload = {
                    "type": "citation",
                    "chunk_id": citation.chunk_id,
                    "filename": citation.filename,
                    "section_path": citation.section_path,
                }
                yield _sse_event(payload)

            yield _sse_event({"type": "done"})

        except LLMUnavailableError as exc:
            yield _sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in chat stream")
            yield _sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_store
from api.schemas import ChatRequest, ValidationErrorResponse
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from rag.prompt import build_messages
from rag.retriever import retrieve
from store.base import VectorStore

router = APIRouter()


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
                            "Newline-delimited SSE stream. Each event is a JSON object. "  # noqa: E501
                            "See TokenEvent, CitationEvent, DoneEvent, ErrorEvent schemas."  # noqa: E501
                        ),
                    },
                    "examples": {
                        "token": {
                            "summary": "Token event",
                            "value": 'data: {"type": "token", "content": "The main"}\n\n',  # noqa: E501
                        },
                        "citation": {
                            "summary": "Citation event",
                            "value": (
                                'data: {"type": "citation", "chunk_id": "chunk-1",'
                                ' "filename": "retro.md", "section_path": "Retro > Blockers"}\n\n'  # noqa: E501
                            ),
                        },
                        "done": {
                            "summary": "Done event",
                            "value": 'data: {"type": "done"}\n\n',
                        },
                        "error": {
                            "summary": "Error event",
                            "value": 'data: {"type": "error", "message": "LLM backend unreachable"}\n\n',  # noqa: E501
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
            chunks = retrieve(body.question, llm, store, body.top_k, body.min_score)
            history: list[Message] = [
                Message(role=h.role, content=h.content) for h in body.history
            ]
            messages = build_messages(body.question, chunks, history)

            for token in llm.stream(messages):
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            for citation in build_citations(chunks):
                payload = {
                    "type": "citation",
                    "chunk_id": citation.chunk_id,
                    "filename": citation.filename,
                    "section_path": citation.section_path,
                }
                yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except LLMUnavailableError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

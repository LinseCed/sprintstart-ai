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
                yield f"data: {json.dumps({'type': 'citation', 'chunk_id': citation.chunk_id, 'filename': citation.filename, 'section_path': citation.section_path})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except LLMUnavailableError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

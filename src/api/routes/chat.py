from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agents.orchestrator import ChatOrchestrator
from api.dependencies import get_orchestrator
from api.schemas import ChatRequest, ValidationErrorResponse

router = APIRouter()


@router.post(
    "/chat",
    summary="Ask a question (streaming)",
    response_class=StreamingResponse,
    description=(
        "Runs the agentic pipeline, then streams a generated answer token by "
        "token as Server-Sent Events.\n\n"
        "Event sequence:\n"
        "1. Zero or more `tool_use` events (the agents and tools the orchestrator "
        "invoked, in order)\n"
        "2. Zero or more `token` events (the answer, in order)\n"
        "3. Zero or more `citation` events (sources used)\n"
        "4. Exactly one `done` event\n\n"
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
                            "See ToolUseEvent, TokenEvent, CitationEvent, DoneEvent, ErrorEvent schemas."  # noqa: E501
                        ),
                    },
                    "examples": {
                        "tool_use": {
                            "summary": "Tool use event",
                            "value": 'data: {"type": "tool_use", "name": "retrieve", "kind": "tool"}\n\n',  # noqa: E501
                        },
                        "token": {
                            "summary": "Token event",
                            "value": 'data: {"type": "token", "content": "The main"}\n\n',  # noqa: E501
                        },
                        "citation": {
                            "summary": "Citation event",
                            "value": (
                                'data: {"type": "citation", "chunk_id": "chunk-1",'
                                ' "artifact_id": "artifact-1",'
                                ' "filename": "retro.md", "start_line": 12}\n\n'
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
                "application/json": {"example": {"detail": "'prompt' is required"}}
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
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    return StreamingResponse(
        orchestrator.stream(body.prompt, body.context),
        media_type="text/event-stream",
    )

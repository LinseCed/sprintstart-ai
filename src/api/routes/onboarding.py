from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_onboarding_orchestrator
from api.schemas import OnboardingPathRequest, ValidationErrorResponse
from onboarding.orchestrator import OnboardingOrchestrator

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


@router.post(
    "/path",
    summary="Generate a personalized onboarding path (streaming)",
    response_class=StreamingResponse,
    description=(
        "Runs the deterministic onboarding-path pipeline and streams progress as "
        "Server-Sent Events.\n\n"
        "Event sequence:\n"
        "1. One `stage` event per pipeline stage (select, filter, retrieve, "
        "synthesize, validate, emit)\n"
        "2. Exactly one `path` event (the structured path, its YAML, and the "
        "quality report)\n"
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
                            "object. See StageEvent, PathEvent, DoneEvent, "
                            "ErrorEvent schemas."
                        ),
                    },
                    "examples": {
                        "stage": {
                            "summary": "Stage event",
                            "value": 'data: {"type": "stage", "name": "retrieve"}\n\n',
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
                "application/json": {
                    "example": {"detail": "'working_area' is required"}
                }
            },
        },
    },
)
def onboarding_path(
    body: OnboardingPathRequest,
    orchestrator: OnboardingOrchestrator = Depends(get_onboarding_orchestrator),
) -> StreamingResponse:
    return StreamingResponse(
        orchestrator.stream(body.to_profile()),
        media_type="text/event-stream",
    )

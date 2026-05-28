from fastapi import APIRouter, Depends, Response

from api.dependencies import get_llm
from api.schemas import HealthResponse
from llm.base import LLMClient
from llm.errors import LLMUnavailableError

router = APIRouter()


@router.get(
    "/health",
    responses={
        200: {
            "model": HealthResponse,
            "content": {"application/json": {"example": {"status": "ok"}}},
        },
        503: {
            "model": HealthResponse,
            "content": {
                "application/json": {
                    "example": {
                        "status": "degraded",
                        "detail": "LLM backend unreachable at 'http://localhost:11434'",
                    }
                }
            },
        },
    },
)
def health(response: Response, llm: LLMClient = Depends(get_llm)) -> HealthResponse:
    try:
        llm.embed("ping")
        return HealthResponse(status="ok")
    except LLMUnavailableError as exc:
        response.status_code = 503
        return HealthResponse(status="degraded", detail=str(exc))

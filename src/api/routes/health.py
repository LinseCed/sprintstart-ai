from fastapi import APIRouter

from api.schemas import HealthResponse

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
def health() -> HealthResponse:
    return HealthResponse(status="ok")

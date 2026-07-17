"""Generation API for AI-synthesized competency lessons.

The AI service is stateless: this router runs lesson synthesis for one
(competency, level) pair over the ingested corpus and returns the result. The
backend owns persistence -- it passes the fingerprint it last recorded for
this lesson so an unchanged corpus doesn't regenerate it.
"""

from typing import Annotated, get_args

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_llm, get_store
from api.schemas import SynthesizeLessonRequest, ValidationErrorResponse
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.lesson_models import LessonLevel, LessonOutcome
from onboarding.lessons import synthesize_lesson
from store.base import VectorStore

router = APIRouter(prefix="/onboarding/lessons", tags=["onboarding-lessons"])

_VALID_LEVELS = set(get_args(LessonLevel))


@router.post(
    "/synthesize",
    response_model=LessonOutcome,
    summary="Synthesize a grounded lesson for one competency at one level",
    description=(
        "Runs lesson synthesis for a single (competency, level) pair over the "
        "ingested corpus and returns a grounded lesson for the backend to persist "
        "as a node's 'Learn' content -- never auto-applied without grounding. The "
        "job is idempotent given the caller's last recorded fingerprint for this "
        "exact lesson: an unchanged corpus yields an `unchanged` outcome.\n\n"
        "This is a heavyweight, schedulable operation (one retrieval + LLM pass); "
        "it is not on the onboarding request path."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during generation.",
        }
    },
)
def synthesize(
    request: SynthesizeLessonRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> LessonOutcome:
    if request.level not in _VALID_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Unknown level {request.level!r}; "
                f"expected one of {sorted(_VALID_LEVELS)}"
            ),
        )
    try:
        return synthesize_lesson(
            llm,
            store,
            competency_key=request.competency_key,
            competency_label=request.competency_label,
            competency_description=request.competency_description,
            level=request.level,  # type: ignore[arg-type]
            last_fingerprint=request.last_fingerprint,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lesson synthesis failed: {exc}",
        ) from exc

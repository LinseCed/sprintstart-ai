"""Assembly API for task-scoped orientation packets.

The AI service is stateless: this router assembles one packet for one task over
the ingested corpus and returns it. The backend owns caching against the task
and the corpus fingerprint it was built from — there is no approval lifecycle
here, deliberately. A packet is disposable, so nobody stands between a hire and
their orientation.

Unlike module proposal this *is* on a hire's request path (first read of a task
they just claimed), which is why the backend caches the result rather than
re-assembling per view.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_llm, get_store
from api.schemas import AssembleOrientationRequest, ValidationErrorResponse
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.orientation import assemble_orientation
from onboarding.orientation_models import OrientationOutcome
from store.base import VectorStore

router = APIRouter(prefix="/onboarding/orientation", tags=["onboarding-orientation"])


@router.post(
    "",
    response_model=OrientationOutcome,
    summary="Assemble a task-scoped orientation packet",
    description=(
        "Assembles what the project's own material already says about doing one "
        "task, segmented by step (set up, find the code, make the change, check "
        "locally, open the PR). Nothing is authored: every section carries the "
        "chunks it came from, and a section that cites nothing is dropped.\n\n"
        "A packet is disposable and needs no approval. An empty corpus, no "
        "retrieved evidence, or a packet whose every section was ungrounded all "
        "return `skipped` with no packet -- the caller must show that as an "
        "honest empty state and never as guidance."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during assembly.",
        }
    },
)
def assemble(
    request: AssembleOrientationRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> OrientationOutcome:
    if not request.task_title.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="task_title must not be empty; a packet is scoped to a task.",
        )
    try:
        return assemble_orientation(
            llm,
            store,
            task_title=request.task_title.strip(),
            task_body=request.task_body,
            labels=request.labels,
            touched_paths=request.touched_paths,
            last_fingerprint=request.last_fingerprint,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orientation assembly failed: {exc}",
        ) from exc

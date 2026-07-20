"""Generation API for AI-proposed competency graph elements.

The AI service is stateless: this router runs the batch proposal job over the
ingested corpus and returns candidate competencies/edges. The backend owns
persistence and PM review — it passes its current live graph in on every
request so proposals can be deduplicated against what already exists.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_llm, get_store
from api.schemas import (
    GenerateCompetencyGraphRequest,
    GraphProposalOutcomeSchema,
    ValidationErrorResponse,
)
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.graph_generation import generate_competency_graph
from store.base import VectorStore

router = APIRouter(
    prefix="/onboarding/competency-graph", tags=["onboarding-competency-graph"]
)


@router.post(
    "/propose",
    response_model=GraphProposalOutcomeSchema,
    summary="Propose competency graph nodes/edges from the corpus",
    description=(
        "Runs the batch proposal job over the ingested corpus and returns candidate "
        "`SKILL`/`CONCEPT` competencies plus the `PREREQUISITE`/`RELATED` edges "
        "between them, for the backend to persist as proposals awaiting PM approval "
        "-- never auto-applied.\n\n"
        "Nodes and relationships are two separate passes. The caller's last recorded "
        "fingerprint makes the *node* pass idempotent (an unchanged corpus proposes "
        "no new nodes), but the relationships pass still runs: the structure between "
        "nodes already in the graph is not a function of the corpus, so a sparse "
        "graph can be re-densified without re-ingesting anything.\n\n"
        "This is a heavyweight, schedulable operation (one retrieval + two LLM passes "
        "over the whole corpus); it is not on the onboarding request path."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during generation.",
        }
    },
)
def propose(
    request: GenerateCompetencyGraphRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> GraphProposalOutcomeSchema:
    try:
        outcome = generate_competency_graph(
            llm,
            store,
            active_competencies=[c.to_model() for c in request.active_competencies],
            active_edges=[e.to_model() for e in request.active_edges],
            last_fingerprint=request.last_fingerprint,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Competency graph proposal failed: {exc}",
        ) from exc
    return GraphProposalOutcomeSchema(**outcome.model_dump())

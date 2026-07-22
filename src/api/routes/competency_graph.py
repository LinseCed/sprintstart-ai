"""Generation API for AI-proposed competency graph elements.

The AI service is stateless: this router runs the batch proposal job over the
ingested corpus and returns candidate competencies/edges. The backend owns
persistence and PM review — it passes its current live graph in on every
request so proposals can be deduplicated against what already exists.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_store
from api.schemas import (
    GenerateCompetencyGraphRequest,
    GraphProposalOutcomeSchema,
    ValidationErrorResponse,
)
from api.sse import stream_progress
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.graph_generation import (
    generate_competency_graph,
    stream_competency_graph,
)
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


@router.post(
    "/propose/stream",
    response_class=StreamingResponse,
    summary="Propose competency graph nodes/edges from the corpus (streaming)",
    description=(
        "The same proposal job as `POST /onboarding/competency-graph/propose`, "
        "streamed as Server-Sent Events so a PM can watch the graph assemble: a "
        "`stage` per pass (retrieving → grounding → linking), an `item` per "
        "competency as it clears grounding and then per accepted edge, and a "
        "terminal `done` carrying the whole outcome. The `done` result is "
        "identical to what the non-streaming endpoint returns -- the stream is a "
        "view of the same computation, never a second answer. An LLM outage "
        "arrives as a terminal `error` event, not an HTTP error."
    ),
    responses={422: {"model": ValidationErrorResponse}},
)
def propose_stream(
    request: GenerateCompetencyGraphRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> StreamingResponse:
    events = stream_competency_graph(
        llm,
        store,
        active_competencies=[c.to_model() for c in request.active_competencies],
        active_edges=[e.to_model() for e in request.active_edges],
        last_fingerprint=request.last_fingerprint,
    )
    return StreamingResponse(
        stream_progress(events, operation="competency_graph"),
        media_type="text/event-stream",
    )

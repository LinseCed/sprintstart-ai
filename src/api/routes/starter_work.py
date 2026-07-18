"""Starter-work pool APIs: mining candidates from open issues, and hire fit ranking.

The AI service is stateless: ``/mine`` runs the batch mining job over the
ingested corpus's open GitHub issues and returns candidate starter tasks for
the backend to persist as proposals awaiting PM approval -- never
auto-applied, mirroring ``api/routes/competency_graph.py``. ``/match`` ranks
an already-approved pool against one hire's competencies; it makes no LLM
generation call (only embeddings), so it has no 503 case tied to LLM
unavailability the way generation endpoints do.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    MatchHireToPoolRequest,
    MineStarterWorkRequest,
    ValidationErrorResponse,
)
from ingestion.metadata_store import IngestionMetadataStore
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.matching import HireCompetency, RankedStarterTask, match_hire_to_pool
from onboarding.starter_work import StarterWorkOutcome, generate_starter_work_pool
from store.base import VectorStore

router = APIRouter(prefix="/onboarding/starter-work", tags=["onboarding-starter-work"])


@router.post(
    "/mine",
    response_model=StarterWorkOutcome,
    summary="Mine open GitHub issues for starter-work pool candidates",
    description=(
        "Runs the batch mining job over the ingested corpus's open GitHub issues "
        "and returns candidate starter tasks for the backend to persist as "
        "proposals awaiting PM approval -- never auto-applied. Only issues with "
        "state OPEN are ever considered; closed issues are excluded "
        "deterministically before the LLM sees them. Idempotent given the "
        "caller's last recorded fingerprint: an unchanged corpus yields an "
        "`unchanged` outcome.\n\n"
        "This is a heavyweight, schedulable operation; it is not on the hire's "
        "request path."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during generation.",
        }
    },
)
def mine(
    request: MineStarterWorkRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
    metadata_store: Annotated[
        IngestionMetadataStore, Depends(get_ingestion_metadata_store)
    ],
) -> StarterWorkOutcome:
    try:
        return generate_starter_work_pool(
            llm,
            store,
            metadata_store,
            active_source_ids=request.active_source_ids,
            active_competency_keys=request.active_competency_keys,
            last_fingerprint=request.last_fingerprint,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Starter-work mining failed: {exc}",
        ) from exc


@router.post(
    "/match",
    response_model=list[RankedStarterTask],
    summary="Rank the starter-work pool by fit against one hire's competencies",
    description=(
        "Ranks the given (already PM-approved) pool by fit against a hire's "
        "freshly-built competencies. Deterministic: competency-key overlap is "
        "the primary score, with embedding similarity only breaking ties -- no "
        "LLM generation call is made, only embeddings."
    ),
)
def match(
    request: MatchHireToPoolRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> list[RankedStarterTask]:
    hire_competencies = [
        HireCompetency(key=c.key, label=c.label, description=c.description)
        for c in request.hire_competencies
    ]
    pool = [t.to_model() for t in request.pool]
    return match_hire_to_pool(llm, hire_competencies, pool)

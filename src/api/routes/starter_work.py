"""Starter-work pool API: mining candidate starter tasks from open issues.

The AI service is stateless: ``/mine`` runs the batch mining job over the
ingested corpus's open GitHub issues and returns candidate starter tasks for
the backend to persist as proposals awaiting PM approval -- never
auto-applied, mirroring ``api/routes/competency_graph.py``.

Hire-to-pool ranking (the old ``/match``) has been retired: it moved into the
backend in slice 4, because a hire is owed a plain-language reason a task fits
and an embedding tie-break cannot give one (#32).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    MineStarterWorkRequest,
    ValidationErrorResponse,
)
from ingestion.metadata_store import IngestionMetadataStore
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
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

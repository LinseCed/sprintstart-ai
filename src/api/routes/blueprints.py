"""Review-queue API for AI-proposed onboarding blueprints (issue #110).

Generation drafts are never activated directly: this router exposes the human
governance layer over the file-based review queue (``onboarding/drafts.py``) —
trigger generation, list pending drafts, inspect the diff against the active
blueprint, approve (promote + retain prior version), discard, and roll back.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.dependencies import get_llm, get_store
from api.schemas import (
    GenerateBlueprintsRequest,
    RollbackBlueprintRequest,
    ValidationErrorResponse,
)
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding import drafts
from onboarding.drafts import BlueprintDiff
from onboarding.generation import GenerationOutcome, generate_blueprints
from onboarding.models import Blueprint
from store.base import VectorStore

router = APIRouter(prefix="/onboarding/blueprints", tags=["onboarding-blueprints"])


class GenerateResponse(BaseModel):
    outcomes: list[GenerationOutcome]


class DraftSummary(BaseModel):
    blueprint: Blueprint
    diff: BlueprintDiff


class DraftListResponse(BaseModel):
    items: list[DraftSummary]


class VersionListResponse(BaseModel):
    scope: str
    versions: list[str]


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Draft/update blueprints from the corpus",
    description=(
        "Runs the batch generation job over the ingested corpus and writes "
        "`source: generated` drafts to the review queue. Drafts are NOT "
        "activated — promotion requires approval. The job is idempotent: an "
        "unchanged corpus produces no new drafts.\n\n"
        "This is a heavyweight, schedulable operation (one retrieval + LLM pass "
        "per scope); it is not on the onboarding request path."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during generation.",
        }
    },
)
def generate(
    request: GenerateBlueprintsRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> GenerateResponse:
    try:
        outcomes = generate_blueprints(llm, store, scopes=request.scopes)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blueprint generation failed: {exc}",
        ) from exc
    return GenerateResponse(outcomes=outcomes)


@router.get(
    "/drafts",
    response_model=DraftListResponse,
    summary="List pending blueprint drafts",
    description="Returns every draft awaiting review with its diff against active.",
)
def list_drafts() -> DraftListResponse:
    items = [
        DraftSummary(blueprint=draft, diff=drafts.diff_against_active(draft))
        for draft in drafts.list_drafts()
    ]
    return DraftListResponse(items=items)


@router.get(
    "/drafts/{scope}/diff",
    response_model=BlueprintDiff,
    summary="Diff a draft against the active blueprint",
    responses={
        404: {"model": ValidationErrorResponse, "description": "No draft for scope."}
    },
)
def draft_diff(scope: str) -> BlueprintDiff:
    draft = drafts.get_draft(scope)
    if draft is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No draft for scope {scope!r}.",
        )
    return drafts.diff_against_active(draft)


@router.post(
    "/drafts/{scope}/approve",
    response_model=Blueprint,
    summary="Approve a draft (promote to active)",
    description=(
        "Promotes the draft to the active blueprint, retaining the outgoing "
        "version for rollback. This is the human-approval gate."
    ),
    responses={
        404: {"model": ValidationErrorResponse, "description": "No draft for scope."}
    },
)
def approve_draft(scope: str) -> Blueprint:
    try:
        return drafts.approve_draft(scope)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.delete(
    "/drafts/{scope}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Discard a draft",
    responses={
        404: {"model": ValidationErrorResponse, "description": "No draft for scope."}
    },
)
def discard_draft(scope: str) -> None:
    if not drafts.discard_draft(scope):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No draft for scope {scope!r}.",
        )


@router.get(
    "/{scope}/versions",
    response_model=VersionListResponse,
    summary="List retained blueprint versions",
)
def list_versions(scope: str) -> VersionListResponse:
    return VersionListResponse(scope=scope, versions=drafts.list_versions(scope))


@router.post(
    "/{scope}/rollback",
    response_model=Blueprint,
    summary="Roll back to a retained version",
    description="Restores a previously retained version as the active blueprint.",
    responses={
        404: {"model": ValidationErrorResponse, "description": "No such version."}
    },
)
def rollback(scope: str, request: RollbackBlueprintRequest) -> Blueprint:
    try:
        return drafts.rollback(scope, request.version)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

"""Domain models for AI-synthesized competency lessons.

A lesson is the "Learn" zone of a graph node's Frame -> Learn -> Verify ->
Payoff module (see backend issue #8): grounded teaching content for one
competency at one target level. Like :mod:`onboarding.graph_generation`, this
service never persists lessons -- it synthesizes one per request and the
backend stores/serves it, memoizing regeneration via the caller-owned
``last_fingerprint`` the same way competency graph proposals do.
"""

from typing import Literal

from pydantic import BaseModel, Field

from onboarding.models import CitationRef

# Mirrors ``onboarding.models.SKILL_LEVELS`` so "level" means the same thing
# for lesson depth as it does for blueprint personalization and the backend's
# ``SkillLevel``/frontend's ``SkillLevel`` union.
LessonLevel = Literal["beginner", "intermediate", "advanced", "expert"]

LessonStatus = Literal["synthesized", "unchanged", "skipped"]


class LessonContent(BaseModel):
    """A grounded lesson for one (competency, level) pair."""

    competency_key: str
    level: LessonLevel
    title: str
    body: str = Field(description="Grounded lesson body, markdown.")
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class LessonProvenance(BaseModel):
    """Why a lesson looks the way it does; mirrors ``GraphProvenance``."""

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class LessonOutcome(BaseModel):
    """Result of one lesson synthesis run."""

    status: LessonStatus
    lesson: LessonContent | None = None
    provenance: LessonProvenance | None = None
    chunks_retrieved: int = 0
    notes: list[str] = Field(default_factory=list[str])

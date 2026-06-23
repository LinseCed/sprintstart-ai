"""Domain models for the personalized onboarding-path generator.

Blueprints are curated, versioned building blocks scoped by ``global`` or
``area:<name>``. Experience is *not* a blueprint axis; it is carried as
step-level metadata (``min_experience`` / ``audience``) plus a tuning signal for
the LLM personalization layer. The :class:`PersonProfile` is intentionally
extensible (``skills`` / ``tags``) so a future multi-dimensional skill profile
slots in without reshaping the API or the blueprint step model.
"""

from typing import Literal

import yaml
from pydantic import BaseModel, Field

Requirement = Literal["required", "recommended"]
Origin = Literal["blueprint", "llm"]
Source = Literal["authored", "generated"]

# Coarse, ordinal experience levels used to gate steps by ``min_experience``.
# Unknown values rank as 0 (most inclusive) so unseen experience never crashes
# and required steps are re-injected by the coverage gate regardless.
EXPERIENCE_LEVELS: dict[str, int] = {
    "junior": 1,
    "mid": 2,
    "intermediate": 2,
    "senior": 3,
    "lead": 4,
}


def experience_rank(level: str | None) -> int:
    """Ordinal rank for a coarse experience level; unknown/None -> 0."""
    if level is None:
        return 0
    return EXPERIENCE_LEVELS.get(level.strip().lower(), 0)


class PersonProfile(BaseModel):
    """Who the path is generated for.

    ``experience`` is a coarse level today; ``skills``/``tags`` make the input
    forward-compatible with a richer, LLM-derived skill evaluation later.
    """

    working_area: str = Field(description="e.g. backend, frontend, devops")
    experience: str = Field(description="Coarse experience level, e.g. junior")
    skills: list[str] = Field(default_factory=list[str])
    tags: list[str] = Field(default_factory=list[str])


class Resource(BaseModel):
    """An authored hint pointing at a document; optional."""

    filename: str
    note: str | None = None


class CitationRef(BaseModel):
    """A resolved reference to an ingested document chunk."""

    filename: str
    chunk_id: str


class BlueprintStep(BaseModel):
    id: str
    title: str
    description: str = ""
    requirement: Requirement = "recommended"
    audience: list[str] = Field(default_factory=list[str])
    min_experience: str | None = None
    tags: list[str] = Field(default_factory=list[str])
    resources: list[Resource] = Field(default_factory=list[Resource])


class Blueprint(BaseModel):
    """A versioned, scoped set of onboarding steps.

    ``source`` + ``version`` give provenance/rollback. ``source`` is the
    pluggable seam: this increment ships ``authored``; ``generated`` (issue
    #110) reuses the identical serve path.
    """

    scope: str = Field(description="'global' or 'area:<name>'")
    version: str = "0"
    source: Source = "authored"
    steps: list[BlueprintStep] = Field(default_factory=list[BlueprintStep])


class PathStep(BaseModel):
    id: str
    title: str
    description: str = ""
    requirement: Requirement = "recommended"
    origin: Origin = "blueprint"
    tags: list[str] = Field(default_factory=list[str])
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class PathPhase(BaseModel):
    title: str
    steps: list[PathStep] = Field(default_factory=list[PathStep])


class QualityReport(BaseModel):
    """Deterministic rubric over the assembled path; recorded for regression."""

    coverage: float = Field(description="required steps present / expected")
    grounded_ratio: float = Field(description="LLM steps cited / LLM steps")
    ordering_valid: bool
    score: float
    notes: list[str] = Field(default_factory=list[str])


class OnboardingPath(BaseModel):
    working_area: str
    experience: str
    phases: list[PathPhase] = Field(default_factory=list[PathPhase])
    # Identifiable versions so onboarding outcomes can later be attributed.
    blueprint_versions: dict[str, str] = Field(default_factory=dict[str, str])
    quality: QualityReport

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)

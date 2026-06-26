"""Domain models for the personalized onboarding-path generator.

Blueprints are curated, versioned building blocks scoped by ``global`` or
``area:<name>``. Experience is *not* a blueprint axis; it is carried as
step-level metadata (``min_experience`` / ``audience``) plus a tuning signal for
the LLM personalization layer. The :class:`PersonProfile` is intentionally
extensible (``skills`` / ``tags``) so a future multi-dimensional skill profile
slots in without reshaping the API or the blueprint step model.
"""

import hashlib
import re
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Requirement = Literal["required", "recommended"]
Origin = Literal["blueprint", "llm"]
Source = Literal["authored", "generated"]

# Coarse, ordinal experience levels used to gate steps by ``min_experience`` and
# to tune the synthesis verbosity. Single source of truth for both, so the two
# consumers can't disagree about what a level means. Synonyms map to the same
# rank. Unknown values rank as 0 (most inclusive) so unseen experience never
# crashes and required steps are re-injected by the coverage gate regardless.
EXPERIENCE_LEVELS: dict[str, int] = {
    "intern": 1,
    "entry": 1,
    "junior": 1,
    "mid": 2,
    "intermediate": 2,
    "senior": 3,
    "lead": 4,
    "staff": 4,
    "principal": 5,
}


def experience_rank(level: str | None) -> int:
    """Ordinal rank for a coarse experience level; unknown/None -> 0."""
    if level is None:
        return 0
    return EXPERIENCE_LEVELS.get(level.strip().lower(), 0)


def content_id(title: str) -> str:
    """Content fingerprint of a step title: ``step-<8 hex>``.

    Used two ways: as a step's **id at birth** (assigned once, then frozen and
    stored — so renaming the title keeps the step's identity and history) and as
    the **fingerprint** for write-time de-duplication (two steps with the same
    normalized title share a fingerprint and collapse to one record).
    Normalization is case- and whitespace-insensitive.
    """
    normalized = re.sub(r"\s+", " ", title).strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"step-{digest[:8]}"


class PersonProfile(BaseModel):
    """Who the path is generated for.

    ``experience`` is a coarse level today; ``skills``/``tags`` make the input
    forward-compatible with a richer, LLM-derived skill evaluation later.
    """

    working_area: str = Field(description="e.g. backend, frontend, devops")
    experience: str = Field(description="Coarse experience level, e.g. junior")
    skills: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Resource(BaseModel):
    """An authored hint pointing at a document; optional."""

    filename: str
    note: str | None = None


class CitationRef(BaseModel):
    """A resolved reference to an ingested document chunk."""

    filename: str
    chunk_id: str


class Task(BaseModel):
    """An actionable sub-step within an onboarding step."""

    title: str = Field(description="Actionable sub-step title")
    description: str = Field(default="", description="Optional details for this task")


class StepRecord(BaseModel):
    """A unit of onboarding content — the registry's aggregate root.

    Stored in the step pool (``blueprints/steps.yaml``). The ``id`` is assigned
    once at creation as ``content_id(title)`` and then *frozen* — editing the
    title never changes it, so a step keeps its identity (and history) across
    renames. Status (``requirement`` / ``invariant``) is deliberately absent: it
    is structural and lives on the :class:`SkeletonRef` that points at the step,
    so the same step can be required for one area and recommended for another.
    """

    id: str
    title: str
    description: str = ""
    audience: list[str] = Field(default_factory=list)
    min_experience: str | None = None
    tags: list[str] = Field(default_factory=list)
    resources: list[Resource] = []
    # Intrinsic grounding for AI-generated steps (issue #110); authored steps
    # leave this empty and rely on the serve-time enrichment layer instead.
    citations: list[CitationRef] = []


class BlueprintStep(BaseModel):
    """A step as *served*: a :class:`StepRecord`'s content merged with the
    contextual status from the :class:`SkeletonRef` that referenced it.

    This is the resolved view the pipeline, quality gate, diff, and management
    API operate on — unchanged in shape so the registry stays internal.
    """

    id: str
    title: str
    description: str = ""
    requirement: Requirement = "recommended"
    audience: list[str] = Field(default_factory=list)
    min_experience: str | None = None
    tags: list[str] = Field(default_factory=list)
    resources: list[Resource] = []
    citations: list[CitationRef] = []
    tasks: list[Task] = Field(default_factory=list)
    # Human-owned protection flag. An ``invariant`` step may not be removed or
    # downgraded by the generation job; such changes are blocked or escalated.
    invariant: bool = False


class BlueprintProvenance(BaseModel):
    """Why a generated blueprint looks the way it does.

    ``corpus_fingerprint`` ties a generated blueprint to the exact corpus state
    it was drafted from, which is what makes the generation job idempotent:
    re-running against an unchanged corpus produces the same fingerprint and is
    skipped. ``None`` throughout for authored blueprints.
    """

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list)


class Blueprint(BaseModel):
    """A versioned, scoped set of onboarding steps.

    ``source`` + ``version`` give provenance/rollback. ``source`` is the
    pluggable seam: ``authored`` and ``generated`` (issue #110) share the
    identical serve path. ``provenance`` is populated for generated blueprints.
    """

    scope: str = Field(description="'global' or 'area:<name>'")
    version: str = "0"
    source: Source = "authored"
    steps: list[BlueprintStep] = []
    provenance: BlueprintProvenance | None = None


class SkeletonRef(BaseModel):
    """A skeleton's ordered reference to a step in the pool.

    ``requirement`` / ``invariant`` are properties of the step *in this path*,
    not of the step itself, so a step can be required for one scope and merely
    recommended for another.
    """

    id: str
    requirement: Requirement = "recommended"
    invariant: bool = False


class Skeleton(BaseModel):
    """The structural layer: an ordered, versioned selection of steps by scope.

    Replaces the on-disk ``Blueprint``. Resolving a skeleton against the step
    pool yields a :class:`Blueprint` — the unchanged served view. ``source`` +
    ``version`` + ``provenance`` carry the same governance/rollback semantics as
    before, now at the structural layer.
    """

    scope: str = Field(description="'global' or 'area:<name>'")
    version: str = "0"
    source: Source = "authored"
    steps: list[SkeletonRef] = []
    provenance: BlueprintProvenance | None = None


class PathStep(BaseModel):
    id: str
    title: str
    description: str = ""
    requirement: Requirement = "recommended"
    origin: Origin = "blueprint"
    tags: list[str] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)
    citations: list[CitationRef] = []
    tasks: list[Task] = Field(default_factory=list)


class PathPhase(BaseModel):
    title: str
    scope: str | None = Field(default=None, exclude=True)
    steps: list[PathStep] = []


class QualityReport(BaseModel):
    """Deterministic rubric over the assembled path; recorded for regression."""

    coverage: float = Field(
        default=0.0, description="required steps present / expected"
    )
    grounded_ratio: float = Field(
        default=0.0, description="LLM steps cited / LLM steps"
    )
    ordering_valid: bool = False
    score: float = 0.0
    notes: list[str] = Field(default_factory=list)


class OnboardingPath(BaseModel):
    working_area: str
    experience: str
    phases: list[PathPhase] = []
    # Identifiable versions so onboarding outcomes can later be attributed.
    blueprint_versions: dict[str, str] = Field(default_factory=dict)
    quality: QualityReport = Field(default_factory=QualityReport)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)

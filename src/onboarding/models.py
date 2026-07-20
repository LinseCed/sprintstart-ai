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
# Matches the backend's ``CheckQuestionType`` enum constants exactly, so the
# generated check needs no case translation on the consuming side.
CheckQuestionType = Literal["MULTIPLE_CHOICE", "SHORT_TEXT"]

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


# Ordinal skill-proficiency levels, aligned 1:1 with the backend ``SkillLevel``
# enum and the frontend ``SkillLevel`` union so all three services agree on what
# a level means. Matching is case-insensitive; unknown values rank as 0 (treated
# as no demonstrated proficiency) so an unseen level never crashes generation.
SKILL_LEVELS: dict[str, int] = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}


def skill_rank(level: str | None) -> int:
    """Ordinal rank for a skill-proficiency level; unknown/None -> 0."""
    if level is None:
        return 0
    return SKILL_LEVELS.get(level.strip().lower(), 0)


def proficiency_rank(skills: "list[SkillAssessment]") -> int:
    """Overall proficiency for gating/verbosity, derived from a person's skills.

    Taken as the highest skill level held (``0`` when no skills are listed, which
    is the most inclusive — every step surfaces). The :data:`SKILL_LEVELS` scale
    lines up rank-for-rank with :data:`EXPERIENCE_LEVELS`
    (beginner↔junior, intermediate↔mid, advanced↔senior, expert↔lead), so this
    rank is directly comparable to a step's ``min_experience`` rank.
    """
    return max((skill_rank(s.level) for s in skills), default=0)


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


class SkillAssessment(BaseModel):
    """A single assessed skill with a proficiency level.

    ``name`` is a free-form skill tag (e.g. ``kotlin``); ``level`` is one of the
    canonical :data:`SKILL_LEVELS` (``beginner``..``expert``), case-insensitive,
    with unknown values ranking as 0 so an unseen level never crashes generation.
    This mirrors the backend ``SkillAssessmentDto`` / frontend ``UserSkillAssessment``
    so a user's leveled skills survive end to end instead of being flattened away.
    """

    name: str
    level: str = Field(default="beginner", description="beginner..expert")


class PersonProfile(BaseModel):
    """Who the path is generated for.

    ``skills`` carry per-skill proficiency levels — the source of truth for how
    experienced the person is. Step gating and synthesis verbosity derive an
    overall level from them via :func:`proficiency_rank`. ``tags`` keep the input
    forward-compatible with richer targeting.
    """

    working_area: str = Field(description="e.g. backend, frontend, devops")
    skills: list[SkillAssessment] = Field(default_factory=list[SkillAssessment])
    tags: list[str] = Field(default_factory=list)


class Resource(BaseModel):
    """An authored hint pointing at a document; optional."""

    filename: str
    note: str | None = None


class CitationRef(BaseModel):
    """A resolved reference to an ingested document chunk."""

    filename: str
    chunk_id: str
    source_url: str | None = None


class Task(BaseModel):
    """An actionable sub-step within an onboarding step."""

    title: str = Field(description="Actionable sub-step title")
    description: str = Field(default="", description="Optional details for this task")


class BlueprintStep(BaseModel):
    """A step of the per-user onboarding path, as path generation consumes it.

    Transitional: the backend derives these from its baseline's competency
    selection, since path generation has not been rewritten around
    competency-owned modules yet.
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
    tasks: list[Task] = []
    # Human-owned protection flag. An ``invariant`` step may not be removed or
    # downgraded by the generation job; such changes are blocked or escalated.
    invariant: bool = False
    # The competency graph key this step targets, when known. The backend's
    # blueprint->target bridge terminates a project's path in the keys its steps
    # declare here (falling back to all-visible when none carry a key), so a
    # matched key is what lets a per-project path narrow to real targets.
    competency_key: str | None = None


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

    ``source`` distinguishes human-authored from AI-generated blueprints.
    ``provenance`` is populated for generated blueprints and drives idempotency.
    """

    scope: str = Field(description="'global' or 'area:<name>'")
    version: str = "0"
    source: Source = "authored"
    steps: list[BlueprintStep] = []
    provenance: BlueprintProvenance | None = None


class BaselineCompetency(BaseModel):
    """One competency selected into a baseline.

    A baseline is a *selection over the competency graph*, not a list of prose
    steps: which competencies must everyone in a scope reach, and to what level.
    ``target_level`` is ``None`` when the scope has no opinion beyond the
    competency's own bar — the normal case.
    """

    competency_key: str
    target_level: int | None = None
    requirement: Requirement = "recommended"
    # Human-owned protection flag. An ``invariant`` entry may not be removed or
    # downgraded by the generation job; such changes are blocked or escalated.
    invariant: bool = False
    # Why this competency belongs in the baseline, in the proposer's words. Shown
    # to the PM reviewing the proposal; not persisted as part of the selection.
    rationale: str = ""


class Baseline(BaseModel):
    """A versioned, scoped competency selection — the mandatory baseline.

    ``source`` distinguishes human-authored from AI-generated baselines.
    ``provenance`` is populated for generated ones and drives idempotency.
    """

    scope: str = Field(description="'global' or 'area:<name>'")
    version: str = "0"
    source: Source = "authored"
    competencies: list[BaselineCompetency] = []
    provenance: BlueprintProvenance | None = None


class PathStep(BaseModel):
    id: str
    title: str
    description: str = ""
    requirement: Requirement = "recommended"
    origin: Origin = "blueprint"
    #: The competency this step teaches, carried over from the blueprint step it
    #: came from. It is what lets the backend attach a graded check to the step
    #: and, through it, turn the matching graph node into an openable module.
    #: ``None`` for steps the LLM added on top of the blueprint, which have no
    #: competency attached to copy.
    competency_key: str | None = None
    tags: list[str] = Field(default_factory=list)
    resources: list[Resource] = []
    citations: list[CitationRef] = []
    tasks: list[Task] = []


class CheckOption(BaseModel):
    """One answer option of a MULTIPLE_CHOICE check question."""

    position: int
    label: str
    correct: bool = False


class CheckQuestion(BaseModel):
    """One knowledge-check question, grounded in its phase's content.

    ``correct_answer`` is only meaningful for ``SHORT_TEXT`` questions;
    ``options`` is only meaningful for ``MULTIPLE_CHOICE`` ones.
    """

    position: int
    type: CheckQuestionType
    question: str
    explanation: str | None = None
    correct_answer: str | None = None
    options: list[CheckOption] = Field(default_factory=list[CheckOption])


class PhaseCheck(BaseModel):
    """A small knowledge-check quiz for a phase; empty when generation fails.

    Absence of questions (rather than a missing/null ``check``) is the
    degraded state, so consumers never need to null-check the field itself.
    """

    questions: list[CheckQuestion] = Field(default_factory=list[CheckQuestion])


class PathPhase(BaseModel):
    title: str
    scope: str | None = Field(default=None, exclude=True)
    steps: list[PathStep] = []
    check: PhaseCheck = Field(default_factory=PhaseCheck)


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
    phases: list[PathPhase] = []
    # Identifiable versions so onboarding outcomes can later be attributed.
    blueprint_versions: dict[str, str] = Field(default_factory=dict)
    quality: QualityReport = Field(default_factory=QualityReport)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)

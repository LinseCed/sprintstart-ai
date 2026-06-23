"""Deterministic staged pipeline that assembles a personalized onboarding path.

Stages: (1) select blueprints by scope, (2) filter/order steps by experience,
(3) retrieve grounding evidence across the corpus, (4) LLM synthesis of the
personalized layer, (5) validation & quality gates, (6) emit. The pipeline is a
generator that yields :class:`StageProgress` markers (the orchestrator turns
these into SSE events) and returns the final :class:`OnboardingPath`.

Gates guarantee a good path independent of the (current or future) blueprint
source: schema (invalid LLM output -> blueprint-only fallback), coverage
(required steps always present), grounding (LLM steps must cite evidence), and
human-owned invariants (e.g. a security-policy step is always present).
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass

from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.blueprints import load_blueprints, select_blueprints
from onboarding.models import (
    Blueprint,
    BlueprintStep,
    CitationRef,
    OnboardingPath,
    PathPhase,
    PathStep,
    PersonProfile,
    QualityReport,
    experience_rank,
)
from onboarding.quality import evaluate
from onboarding.synthesis import SynthesisError, SynthesisResult, synthesize
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

STAGES = ("select", "filter", "retrieve", "synthesize", "validate", "emit")

# Human-owned invariant: every path must contain an acknowledged security step,
# regardless of blueprint source. Enforced in code, not in blueprint data.
_SECURITY_INVARIANT = PathStep(
    id="security-policy-ack",
    title="Read and acknowledge the security policy",
    description="Review the security policy and record your acknowledgement.",
    requirement="required",
    origin="blueprint",
    tags=["security", "compliance"],
)


@dataclass(frozen=True)
class StageProgress:
    name: str


def _phase_title(scope: str) -> str:
    if scope == "global":
        return "Getting started"
    if scope.startswith("area:"):
        return f"{scope.split(':', 1)[1].capitalize()} essentials"
    return scope


def _step_applies(step: BlueprintStep, profile: PersonProfile) -> bool:
    """Filter a step by audience and experience.

    Required steps always apply (they are guaranteed by the coverage gate);
    recommended steps are gated by ``audience`` and ``min_experience``.
    """
    if step.requirement == "required":
        return True

    if step.audience:
        haystack = {profile.working_area.lower(), *(t.lower() for t in profile.tags)}
        haystack.update(s.lower() for s in profile.skills)
        if not haystack & {a.lower() for a in step.audience}:
            return False

    return experience_rank(profile.experience) >= experience_rank(step.min_experience)


def _order_steps(steps: list[BlueprintStep]) -> list[BlueprintStep]:
    """Required steps before recommended ones, preserving authored order."""
    return [s for s in steps if s.requirement == "required"] + [
        s for s in steps if s.requirement != "required"
    ]


class OnboardingPipeline:
    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        *,
        top_k: int = 8,
        min_score: float = 0.3,
    ) -> None:
        self._llm = llm
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._bm25_cache = BM25IndexCache()

    def run(
        self, profile: PersonProfile
    ) -> Generator[StageProgress, None, OnboardingPath]:
        # (1) select
        yield StageProgress("select")
        selected = select_blueprints(load_blueprints(), profile)
        blueprint_versions = {b.scope: b.version for b in selected}
        required_ids = {
            s.id for b in selected for s in b.steps if s.requirement == "required"
        }

        # (2) filter / order
        yield StageProgress("filter")
        phases, kept_steps = _build_phases(selected, profile)

        # (3) retrieve grounding evidence across the corpus
        yield StageProgress("retrieve")
        chunks = self._retrieve(profile, kept_steps)

        # (4) synthesize the personalized layer (schema gate -> fallback)
        yield StageProgress("synthesize")
        notes: list[str] = []
        try:
            result = synthesize(profile, kept_steps, chunks, self._llm)
        except SynthesisError as exc:
            logger.warning("Synthesis failed, using blueprint-only fallback: %s", exc)
            result = SynthesisResult()
            notes.append("LLM synthesis unavailable; blueprint-only fallback")

        # (5) validate & quality gates
        yield StageProgress("validate")
        _apply_enrichments(phases, result.enrichments)
        notes.extend(_apply_grounding_gate(phases, result.added_steps))
        _enforce_coverage(phases, selected, profile)
        _enforce_invariants(phases)

        # (6) emit
        yield StageProgress("emit")
        path = OnboardingPath(
            working_area=profile.working_area,
            experience=profile.experience,
            phases=phases,
            blueprint_versions=blueprint_versions,
            quality=_PLACEHOLDER_QUALITY,
        )
        path.quality = evaluate(path, required_ids, notes)
        return path

    def _retrieve(
        self, profile: PersonProfile, steps: list[BlueprintStep]
    ) -> list[ScoredChunk]:
        if self._store.count() == 0:
            return []
        titles = "; ".join(s.title for s in steps)
        query = f"{profile.working_area} onboarding: {titles}".strip()
        try:
            return hybrid_retrieve(
                question=query,
                llm=self._llm,
                store=self._store,
                top_k=self._top_k,
                min_score=self._min_score,
                bm25_cache=self._bm25_cache,
            )
        except LLMUnavailableError:
            raise
        except Exception:
            logger.exception("Retrieval failed; continuing without grounding")
            return []


_PLACEHOLDER_QUALITY = QualityReport(
    coverage=0.0, grounded_ratio=0.0, ordering_valid=False, score=0.0
)


def _build_phases(
    blueprints: list[Blueprint], profile: PersonProfile
) -> tuple[list[PathPhase], list[BlueprintStep]]:
    phases: list[PathPhase] = []
    kept_steps: list[BlueprintStep] = []

    for blueprint in blueprints:
        applicable = _order_steps(
            [s for s in blueprint.steps if _step_applies(s, profile)]
        )
        if not applicable:
            continue
        kept_steps.extend(applicable)
        phases.append(
            PathPhase(
                title=_phase_title(blueprint.scope),
                steps=[
                    PathStep(
                        id=s.id,
                        title=s.title,
                        description=s.description,
                        requirement=s.requirement,
                        origin="blueprint",
                        tags=s.tags,
                    )
                    for s in applicable
                ],
            )
        )

    return phases, kept_steps


def _apply_enrichments(
    phases: list[PathPhase], enrichments: dict[str, list[CitationRef]]
) -> None:
    for phase in phases:
        for step in phase.steps:
            refs = enrichments.get(step.id)
            if refs:
                step.citations = refs


def _apply_grounding_gate(
    phases: list[PathPhase], added_steps: list[PathStep]
) -> list[str]:
    """Drop LLM-added steps without a citation; append grounded ones."""
    grounded = [s for s in added_steps if s.citations]
    dropped = len(added_steps) - len(grounded)
    notes: list[str] = []
    if dropped:
        notes.append(f"dropped {dropped} ungrounded LLM step(s)")
    if grounded:
        phases.append(PathPhase(title="Recommended for you", steps=grounded))
    return notes


def _enforce_coverage(
    phases: list[PathPhase], blueprints: list[Blueprint], profile: PersonProfile
) -> None:
    """Guarantee every required blueprint step is present; re-inject if missing."""
    present = {s.id for p in phases for s in p.steps}
    for blueprint in blueprints:
        missing = [
            s
            for s in blueprint.steps
            if s.requirement == "required" and s.id not in present
        ]
        if not missing:
            continue
        target = _phase_for_scope(phases, blueprint.scope)
        for step in missing:
            target.steps.insert(
                0,
                PathStep(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                    requirement="required",
                    origin="blueprint",
                    tags=step.tags,
                ),
            )
            present.add(step.id)


def _phase_for_scope(phases: list[PathPhase], scope: str) -> PathPhase:
    title = _phase_title(scope)
    for phase in phases:
        if phase.title == title:
            return phase
    phase = PathPhase(title=title, steps=[])
    phases.insert(0, phase)
    return phase


def _enforce_invariants(phases: list[PathPhase]) -> None:
    """Human-owned invariants enforced regardless of blueprint source."""
    present = {s.id for p in phases for s in p.steps}
    if _SECURITY_INVARIANT.id in present:
        return
    if not phases:
        phases.append(PathPhase(title="Getting started", steps=[]))
    phases[0].steps.insert(0, _SECURITY_INVARIANT.model_copy(deep=True))

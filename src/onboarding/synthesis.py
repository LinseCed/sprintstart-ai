"""LLM personalization layer.

Given the filtered blueprint steps and per-step grounding evidence retrieved
across the corpus, the LLM (a) rewrites each step's description for the person's
experience and skills, (b) attaches document references to blueprint steps, and
(c) proposes additional project-specific steps. The layer is *additive*: it never
drops required steps (the pipeline's coverage gate enforces that), and every
added step must cite real evidence (the grounding gate enforces that).

The LLM is asked for JSON. Invalid / truncated output raises
:class:`SynthesisError`, which the pipeline catches to fall back to a
blueprint-only path.
"""

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.models import (
    BlueprintStep,
    CitationRef,
    PathStep,
    PersonProfile,
    Task,
    content_id,
    experience_rank,
)
from rag.types import ScoredChunk

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    """Raised when the LLM output cannot be parsed/validated."""


class SynthesisResult(BaseModel):
    # step id -> rewritten description tailored to the person
    rewrites: dict[str, str] = Field(default_factory=dict)
    # step id -> resolved citations the LLM attached to that blueprint step
    enrichments: dict[str, list[CitationRef]] = Field(default_factory=dict)
    # LLM-proposed steps (origin="llm"); ungrounded ones are dropped by the gate
    added_steps: list[PathStep] = []
    # step id -> task suggestions from the LLM
    tasks: dict[str, list[Task]] = Field(default_factory=dict)


class _TaskSuggestion(BaseModel):
    title: str
    description: str = ""


class _StepSynthesis(BaseModel):
    id: str
    rewritten: str = ""
    chunk_ids: list[str] = Field(default_factory=list)


class _AddedStep(BaseModel):
    title: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    tasks: list[dict[str, str]] = []


class _Payload(BaseModel):
    steps: list[_StepSynthesis] = []
    added: list[_AddedStep] = []
    tasks: dict[str, list[_TaskSuggestion]] = Field(default_factory=dict)


def _verbosity(profile: PersonProfile) -> str:
    """Experience tunes how much the LLM expands vs. compresses each step.

    Keyed off the shared :data:`EXPERIENCE_LEVELS` rank so this stays in step
    with the gating logic. Unknown levels (rank 0) get balanced detail.
    """
    rank = experience_rank(profile.experience)
    if rank == 1:  # intern / entry / junior
        return "Explain steps in extra detail; assume little prior context."
    if rank >= 3:  # senior / lead / staff / principal
        return "Keep steps concise; assume strong prior context."
    return "Use a balanced level of detail."


def _build_prompt(
    profile: PersonProfile,
    steps: list[BlueprintStep],
    chunks_per_step: dict[str, list[ScoredChunk]],
) -> list[Message]:
    step_blocks: list[str] = []
    for step in steps:
        chunks = chunks_per_step.get(step.id, [])
        evidence = (
            "\n".join(f"  [{c.id}] ({c.filename}) {c.text}" for c in chunks)
            or "  (no documents retrieved)"
        )
        step_blocks.append(
            f"Step {step.id}: {step.title}\n"
            f"Current description: {step.description or '(none)'}\n"
            f"Evidence:\n{evidence}"
        )

    system = (
        "You personalize a software-team onboarding path for a specific person. "
        "For each step you are given its current description and evidence chunks "
        "retrieved specifically for that step. Each chunk is prefixed with its id "
        "in square brackets.\n\n"
        "Do three things and return STRICT JSON only (no prose, no markdown fences):\n"
        "1. 'steps': for every blueprint step, provide:\n"
        "   - 'rewritten': the description rewritten for this person's experience "
        "and skills. Use only what the evidence supports; keep it concrete.\n"
        "   - 'chunk_ids': the chunk ids that ground this step (may be empty if "
        "no evidence was retrieved for it).\n"
        "2. 'added': extra project-specific steps not already covered by the "
        "blueprint. Every added step MUST cite at least one chunk id; do not "
        "invent sources.\n"
        "3. 'tasks': for each blueprint step id, an optional list of concrete, "
        "actionable sub-step tasks — as many as the step genuinely needs (usually "
        "a handful; add more only when the work truly calls for it). Prefer fewer, "
        "high-signal tasks over padding with filler. Each task has a 'title' and "
        "optional 'description'.\n\n"
        f"{_verbosity(profile)}\n\n"
        'JSON schema: {"steps": [{"id": str, "rewritten": str, "chunk_ids": [str]}], '
        '"added": [{"title": str, "description": str, "tags": [str], '
        '"chunk_ids": [str], "tasks": [{"title": str, "description": str}]}], '
        '"tasks": {"<step_id>": [{"title": str, "description": str}]}}'
    )
    skills = (
        ", ".join(f"{s.name} ({s.level})" for s in profile.skills) or "(none listed)"
    )
    interests = ", ".join(profile.tags) or "(none listed)"
    user = (
        f"Working area: {profile.working_area}\n"
        f"Experience: {profile.experience}\n"
        f"Skills: {skills}\n"
        f"Interests: {interests}\n\n"
        "Blueprint steps:\n\n" + "\n\n".join(step_blocks)
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def synthesize(
    profile: PersonProfile,
    steps: list[BlueprintStep],
    chunks_per_step: dict[str, list[ScoredChunk]],
    llm: LLMClient,
) -> SynthesisResult:
    messages = _build_prompt(profile, steps, chunks_per_step)
    raw = llm.generate(messages)

    try:
        payload = _Payload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise SynthesisError(f"invalid synthesis output: {exc}") from exc

    # Flatten all per-step chunks into one lookup so the LLM can cite any chunk
    # that appeared in any step's evidence pool.
    all_chunks: dict[str, ScoredChunk] = {}
    for chunks in chunks_per_step.values():
        for chunk in chunks:
            all_chunks[chunk.id] = chunk

    valid_ids = {s.id for s in steps}
    rewrites: dict[str, str] = {}
    enrichments: dict[str, list[CitationRef]] = {}
    for item in payload.steps:
        if item.id not in valid_ids:
            continue
        if item.rewritten.strip():
            rewrites[item.id] = item.rewritten
        refs = resolve_citations(item.chunk_ids, all_chunks)
        if refs:
            enrichments[item.id] = refs

    tasks: dict[str, list[Task]] = {}
    for step_id, suggestions in payload.tasks.items():
        if step_id in valid_ids:
            tasks[step_id] = [
                Task(title=t.title, description=t.description) for t in suggestions
            ]

    added_steps = [
        PathStep(
            id=content_id(item.title),
            title=item.title,
            description=item.description,
            requirement="recommended",
            origin="llm",
            tags=item.tags,
            citations=resolve_citations(item.chunk_ids, all_chunks),
            tasks=[
                Task(title=t["title"], description=t.get("description", ""))
                for t in item.tasks
            ],
        )
        for item in payload.added
    ]

    return SynthesisResult(
        rewrites=rewrites,
        enrichments=enrichments,
        added_steps=added_steps,
        tasks=tasks,
    )

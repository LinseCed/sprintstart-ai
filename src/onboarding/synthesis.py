"""LLM personalization layer.

Given the filtered blueprint steps and grounding evidence retrieved across the
corpus, the LLM (a) attaches document references to blueprint steps and (b)
proposes additional, project-specific steps. The layer is *additive*: it never
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
from onboarding.models import (
    BlueprintStep,
    CitationRef,
    PathStep,
    PersonProfile,
    content_id,
)
from rag.types import ScoredChunk

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    """Raised when the LLM output cannot be parsed/validated."""


class SynthesisResult(BaseModel):
    # step id -> resolved citations the LLM attached to that blueprint step
    enrichments: dict[str, list[CitationRef]] = Field(default_factory=dict)
    # LLM-proposed steps (origin="llm"); ungrounded ones are dropped by the gate
    added_steps: list[PathStep] = []


class _Enrichment(BaseModel):
    id: str
    chunk_ids: list[str] = Field(default_factory=list)


class _AddedStep(BaseModel):
    title: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class _Payload(BaseModel):
    enriched: list[_Enrichment] = []
    added: list[_AddedStep] = []


def _verbosity(profile: PersonProfile) -> str:
    """Experience tunes how much the LLM expands vs. compresses each step."""
    level = profile.experience.strip().lower()
    if level in {"junior", "intern", "entry"}:
        return "Explain steps in extra detail; assume little prior context."
    if level in {"senior", "lead", "staff", "principal"}:
        return "Keep steps concise; assume strong prior context."
    return "Use a balanced level of detail."


def _build_prompt(
    profile: PersonProfile,
    steps: list[BlueprintStep],
    chunks: list[ScoredChunk],
) -> list[Message]:
    evidence = (
        "\n".join(f"[{c.id}] ({c.filename}) {c.text}" for c in chunks)
        or "(no documents retrieved)"
    )

    step_lines = "\n".join(f"- {s.id}: {s.title}" for s in steps) or "(none)"

    system = (
        "You personalize a software-team onboarding path. You are given a set of "
        "blueprint steps and evidence snippets from the organization's knowledge "
        "base. Each snippet is prefixed with its chunk id in square brackets.\n\n"
        "Do two things and return STRICT JSON only (no prose, no markdown fences):\n"
        "1. 'enriched': for each blueprint step that the evidence supports, list "
        "the chunk ids that ground it.\n"
        "2. 'added': propose extra project-specific steps that are NOT already "
        "covered by the blueprint, prioritizing steps relevant to the person's "
        "stated skills and interests. Every added step MUST reference at least "
        "one chunk id from the evidence; do not invent sources.\n\n"
        f"{_verbosity(profile)}\n\n"
        'JSON schema: {"enriched": [{"id": str, "chunk_ids": [str]}], '
        '"added": [{"title": str, "description": str, "tags": [str], '
        '"chunk_ids": [str]}]}'
    )
    skills = ", ".join(profile.skills) or "(none listed)"
    interests = ", ".join(profile.tags) or "(none listed)"
    user = (
        f"Working area: {profile.working_area}\n"
        f"Experience: {profile.experience}\n"
        f"Skills: {skills}\n"
        f"Interests: {interests}\n\n"
        f"Blueprint steps:\n{step_lines}\n\n"
        f"Evidence:\n{evidence}"
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def synthesize(
    profile: PersonProfile,
    steps: list[BlueprintStep],
    chunks: list[ScoredChunk],
    llm: LLMClient,
) -> SynthesisResult:
    messages = _build_prompt(profile, steps, chunks)
    raw = llm.generate(messages)

    try:
        payload = _Payload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise SynthesisError(f"invalid synthesis output: {exc}") from exc

    chunks_by_id = {c.id: c for c in chunks}

    def resolve(chunk_ids: list[str]) -> list[CitationRef]:
        refs: list[CitationRef] = []
        seen: set[str] = set()
        for cid in chunk_ids:
            chunk = chunks_by_id.get(cid)
            if chunk is not None and chunk.id not in seen:
                refs.append(CitationRef(filename=chunk.filename, chunk_id=chunk.id))
                seen.add(chunk.id)
        return refs

    valid_ids = {s.id for s in steps}
    enrichments = {
        e.id: resolve(e.chunk_ids) for e in payload.enriched if e.id in valid_ids
    }

    added_steps = [
        PathStep(
            id=content_id(item.title),
            title=item.title,
            description=item.description,
            requirement="recommended",
            origin="llm",
            tags=item.tags,
            citations=resolve(item.chunk_ids),
        )
        for item in payload.added
    ]

    return SynthesisResult(enrichments=enrichments, added_steps=added_steps)

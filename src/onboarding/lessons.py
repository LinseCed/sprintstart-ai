"""Grounded lesson synthesis for one competency graph node's "Learn" zone.

A batch, re-runnable job -- offline/authoring-time, not on the hire's request
path (see backend issue #8's "Lessons themselves are synthesized offline by
#ai-p3; this issue consumes them"). Reuses the same retrieval layer
(:func:`rag.hybrid.hybrid_retrieve`) and idempotency mechanism
(:func:`onboarding.generation.corpus_fingerprint`) as competency graph
proposals, but scoped to a single competency rather than the whole corpus:
the caller (backend) synthesizes one lesson per (competency, level) it needs,
not a corpus-wide batch.
"""

import json
import logging
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.generation import corpus_fingerprint
from onboarding.lesson_models import (
    LessonContent,
    LessonLevel,
    LessonOutcome,
    LessonProvenance,
)
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K = 12
_MIN_SCORE = 0.3

_DEPTH_BY_LEVEL: dict[str, str] = {
    "beginner": "Explain from first principles; assume little prior context "
    "and spell out *why*, not just *how*.",
    "intermediate": "Use a balanced level of detail; assume basic familiarity "
    "with the stack but not this codebase.",
    "advanced": "Keep it concise and focus on this codebase's specific "
    "choices and trade-offs; assume strong general context.",
    "expert": "Be terse; focus only on what's non-obvious or specific to "
    "this codebase's conventions.",
}


class GenerationError(Exception):
    """Raised when the LLM output for a lesson cannot be parsed/validated."""


class _LessonPayload(BaseModel):
    title: str = ""
    body: str = ""
    chunk_ids: list[str] = []


def _evidence_line(chunk: ScoredChunk) -> str:
    meta = chunk.artifact_type or "FILE"
    if chunk.language:
        meta += f"/{chunk.language}"
    return f"  [{chunk.id}] ({chunk.filename} | {meta}) {chunk.text}"


def _build_prompt(
    competency_label: str,
    competency_description: str,
    level: LessonLevel,
    chunks: list[ScoredChunk],
) -> list[Message]:
    evidence = "\n".join(_evidence_line(c) for c in chunks)
    depth = _DEPTH_BY_LEVEL[level]

    system = (
        "You write a short grounded lesson teaching one competency for a "
        "software team's onboarding, for someone proving they've reached a "
        "specific target level. You are given evidence snippets from the "
        "team's own codebase/docs, each prefixed with its chunk id in square "
        "brackets.\n\n"
        "Teach *why* before *how* -- give enough context that the reasoning "
        "behind this codebase's approach makes sense, not just a recipe to "
        "follow. Every factual claim must be grounded in the evidence below; "
        "never invent a detail the evidence doesn't support.\n\n"
        f"Target level: {level}. {depth}\n\n"
        "Return STRICT JSON only (no prose, no markdown fences):\n"
        '{"title": str, "body": str (markdown), "chunk_ids": [str]} '
        "where 'chunk_ids' lists every evidence chunk id the lesson actually "
        "draws on."
    )
    user = (
        f"Competency: {competency_label}\n"
        f"Description: {competency_description or '(none)'}\n\n"
        f"Evidence:\n{evidence}"
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _parse_payload(raw: str) -> _LessonPayload:
    try:
        return _LessonPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid lesson output: {exc}") from exc


def synthesize_lesson(
    llm: LLMClient,
    store: VectorStore,
    *,
    competency_key: str,
    competency_label: str,
    competency_description: str = "",
    level: LessonLevel = "beginner",
    last_fingerprint: str | None = None,
) -> LessonOutcome:
    """Synthesize a grounded lesson for one (competency, level) pair.

    ``last_fingerprint`` is whatever fingerprint the caller recorded from the
    previous synthesis run for this exact (competency, level) -- idempotency
    is per-lesson, not corpus-wide, since the backend synthesizes lessons one
    node at a time as it needs them.
    """
    fingerprint = corpus_fingerprint(store)

    if last_fingerprint is not None and last_fingerprint == fingerprint:
        return LessonOutcome(
            status="unchanged", notes=["corpus unchanged since last synthesis run"]
        )

    if store.count() == 0:
        return LessonOutcome(status="skipped", notes=["corpus is empty"])

    query = f"{competency_label}: {competency_description}".strip(": ")
    chunks = hybrid_retrieve(
        question=query,
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=BM25IndexCache(),
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
    )
    if not chunks:
        return LessonOutcome(
            status="skipped", notes=["no grounding evidence retrieved"]
        )

    raw = llm.generate(
        _build_prompt(competency_label, competency_description, level, chunks)
    )
    try:
        payload = _parse_payload(raw)
    except GenerationError as exc:
        logger.warning(
            "Lesson synthesis failed for competency %r: %s", competency_key, exc
        )
        return LessonOutcome(
            status="skipped", chunks_retrieved=len(chunks), notes=[str(exc)]
        )

    chunks_by_id = {c.id: c for c in chunks}
    citations = resolve_citations(payload.chunk_ids, chunks_by_id)
    if not citations or not payload.body.strip():
        return LessonOutcome(
            status="skipped",
            chunks_retrieved=len(chunks),
            notes=["no grounded citations in generated lesson"],
        )

    return LessonOutcome(
        status="synthesized",
        lesson=LessonContent(
            competency_key=competency_key,
            level=level,
            title=payload.title or competency_label,
            body=payload.body,
            citations=citations,
        ),
        provenance=LessonProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
        ),
        chunks_retrieved=len(chunks),
    )

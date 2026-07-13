"""LLM-generated phase-level knowledge checks.

Each phase gets a small multiple-choice / short-text quiz grounded in that
phase's step content and evidence, mirroring the backend's check-question
shape 1:1 so no field translation is needed on the consuming side. Generation
is best-effort: unparseable/invalid LLM output degrades to an empty check
(``PhaseCheck()``) rather than breaking path assembly, mirroring the
pipeline's existing schema-gate fallback for step synthesis. An unreachable
LLM is not degraded here — it propagates like any other stage, since the rest
of path generation would fail on the same outage anyway.
"""

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.models import CheckOption, CheckQuestion, PathPhase, PhaseCheck
from rag.types import ScoredChunk

logger = logging.getLogger(__name__)

MIN_QUESTIONS = 3
MAX_QUESTIONS = 5


class _OptionPayload(BaseModel):
    label: str = ""
    correct: bool = False


class _QuestionPayload(BaseModel):
    type: str = ""
    question: str = ""
    explanation: str | None = None
    correct_answer: str | None = None
    options: list[_OptionPayload] = Field(default_factory=list[_OptionPayload])


class _Payload(BaseModel):
    questions: list[_QuestionPayload] = Field(default_factory=list[_QuestionPayload])


def _evidence_line(chunk: ScoredChunk) -> str:
    meta = chunk.artifact_type or "FILE"
    if chunk.language:
        meta += f"/{chunk.language}"
    return f"  ({chunk.filename} | {meta}) {chunk.text}"


def _build_prompt(phase: PathPhase, chunks: list[ScoredChunk]) -> list[Message]:
    step_lines = "\n".join(
        f"- {s.title}: {s.description}" if s.description else f"- {s.title}"
        for s in phase.steps
    )
    evidence = (
        "\n".join(_evidence_line(c) for c in chunks) or "  (no documents retrieved)"
    )

    system = (
        "You write a short knowledge-check quiz for one phase of a software-team "
        "onboarding path, grounded ONLY in the given phase content and evidence.\n\n"
        f"Write {MIN_QUESTIONS}-{MAX_QUESTIONS} questions, mixing MULTIPLE_CHOICE "
        "and SHORT_TEXT types. Rules:\n"
        "- MULTIPLE_CHOICE: at least 2 options, at least 1 marked correct (more "
        "than one may be correct).\n"
        "- SHORT_TEXT: provide a concise, non-empty 'correct_answer' reference.\n"
        "- Every question needs a short 'explanation' of why the answer is "
        "correct, for learning effect.\n"
        "- Base every question strictly on the phase content/evidence below; "
        "never invent facts not supported by it.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences):\n"
        '{"questions": [{"type": "MULTIPLE_CHOICE", "question": str, '
        '"explanation": str, "options": [{"label": str, "correct": bool}]}, '
        '{"type": "SHORT_TEXT", "question": str, "explanation": str, '
        '"correct_answer": str}]}'
    )
    user = f"Phase: {phase.title}\n\nSteps:\n{step_lines}\n\nEvidence:\n{evidence}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _validate_question(item: _QuestionPayload) -> CheckQuestion | None:
    """Drop a question that doesn't meet the backend's hard constraints.

    Positions are assigned later, over the surviving questions only, so a
    dropped question never leaves a gap.
    """
    if not item.question.strip():
        return None

    if item.type == "MULTIPLE_CHOICE":
        options = [o for o in item.options if o.label.strip()]
        if len(options) < 2 or not any(o.correct for o in options):
            return None
        return CheckQuestion(
            position=0,
            type="MULTIPLE_CHOICE",
            question=item.question,
            explanation=item.explanation,
            options=[
                CheckOption(position=i, label=o.label, correct=o.correct)
                for i, o in enumerate(options)
            ],
        )

    if item.type == "SHORT_TEXT":
        if not item.correct_answer or not item.correct_answer.strip():
            return None
        return CheckQuestion(
            position=0,
            type="SHORT_TEXT",
            question=item.question,
            explanation=item.explanation,
            correct_answer=item.correct_answer,
        )

    return None


def generate_phase_check(
    phase: PathPhase, chunks: list[ScoredChunk], llm: LLMClient
) -> PhaseCheck:
    """Generate a knowledge check for one phase; empty check on any soft failure.

    A phase with no steps never gets a check. Malformed/invalid LLM output
    (bad JSON, questions failing validation) degrades to an empty
    :class:`PhaseCheck` rather than raising, so a bad quiz never blocks path
    assembly. ``LLMUnavailableError`` is not caught here — it propagates like
    any other pipeline stage, since path generation as a whole depends on the
    same LLM being reachable.
    """
    if not phase.steps:
        return PhaseCheck()

    raw = llm.generate(_build_prompt(phase, chunks))

    try:
        payload = _Payload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Check generation failed for phase %r: %s", phase.title, exc)
        return PhaseCheck()

    candidates = (_validate_question(item) for item in payload.questions)
    questions = [q for q in candidates if q is not None]
    for position, question in enumerate(questions):
        question.position = position

    return PhaseCheck(questions=questions)

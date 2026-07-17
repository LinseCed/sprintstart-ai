"""Grading for a graph node's "Verify" zone.

Unlike lesson synthesis, this is on the hire's request path -- the backend
calls it synchronously per verification attempt (see backend issue #8's
"grading orchestration calling the AI graders") -- so ``grade_knowledge`` is a
single LLM call and ``grade_exact``/``grade_attest`` make none at all.
``artifact`` grading (real repo-state detection) is out of scope here,
deferred to ai/backend Phase 4.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from llm.base import LLMClient, Message
from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

GradingType = Literal["knowledge", "exact", "attest"]


class GradeResult(BaseModel):
    """Outcome of one verification attempt."""

    passed: bool
    score: float = 0.0
    feedback: str = ""
    hint: str | None = Field(default=None, description="Populated only on fail.")


class _JudgePayload(BaseModel):
    passed: bool = False
    score: float = 0.0
    feedback: str = ""
    hint: str | None = None


def grade_exact(*, canonical_answer: str, answer: str) -> GradeResult:
    """Normalized (case/whitespace-insensitive) exact match; no LLM call."""
    normalized_answer = " ".join(answer.split()).strip().lower()
    normalized_canonical = " ".join(canonical_answer.split()).strip().lower()
    if normalized_answer and normalized_answer == normalized_canonical:
        return GradeResult(passed=True, score=1.0, feedback="Matches exactly.")
    return GradeResult(
        passed=False,
        score=0.0,
        feedback="Does not match the expected answer.",
        hint="Check the exact wording expected for this step.",
    )


def grade_attest(*, answer: str) -> GradeResult:
    """Self-confirmation: a non-blank answer is logged as passed, not judged."""
    if answer.strip():
        return GradeResult(passed=True, score=1.0, feedback="Self-attested.")
    return GradeResult(
        passed=False, score=0.0, feedback="No attestation submitted.", hint=None
    )


_HINT_ESCALATION = {
    1: "Give a gentle nudge toward the right area -- do not name the concept outright.",
    2: "Point at the specific concept or piece of evidence the answer is missing.",
    3: "Be nearly explicit about what the correct reasoning is, short of "
    "stating the rubric answer verbatim.",
}


def _hint_instruction(attempt_no: int) -> str:
    return _HINT_ESCALATION.get(attempt_no, _HINT_ESCALATION[3])


def _build_prompt(
    question: str, rubric: str, evidence: str, answer: str, attempt_no: int
) -> list[Message]:
    system = (
        "You grade a free-text answer to an onboarding verification question, "
        "judging it against the rubric using ONLY the given grounded evidence -- "
        "if the rubric implies a claim the evidence doesn't support, do not hold "
        "the learner to it.\n\n"
        "- Judge the core reasoning/meaning, not exact wording. Paraphrases are "
        "fine as long as the key idea is right.\n"
        "- 'score' is 0..1, how completely the answer satisfies the rubric.\n"
        "- 'passed' is true only if the answer demonstrates the core "
        "understanding the rubric asks for.\n"
        "- 'feedback' is one or two short sentences explaining the verdict.\n"
        "- If 'passed' is false, include a 'hint' for the learner's next "
        "attempt; otherwise 'hint' is null. "
        f"This is attempt {attempt_no}: {_hint_instruction(attempt_no)}\n\n"
        "Return STRICT JSON only (no prose, no markdown fences):\n"
        '{"passed": bool, "score": number, "feedback": str, "hint": str|null}'
    )
    user = (
        f"Question: {question}\n\nRubric: {rubric}\n\nGrounded evidence:\n"
        f"{evidence or '(none)'}\n\nLearner's answer: {answer}"
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def grade_knowledge(
    llm: LLMClient,
    *,
    question: str,
    rubric: str,
    evidence: str,
    answer: str,
    attempt_no: int = 1,
) -> GradeResult:
    """LLM-judge a free-text answer against a rubric and its grounded evidence.

    A blank answer is marked incorrect without an LLM call, mirroring
    ``api/routes/grading.py``'s ``/grade-answers``. Unparseable LLM output
    degrades to a failed, ungraded result rather than raising --
    ``LLMUnavailableError`` is the one exception that propagates, since
    verification as a whole depends on the same LLM being reachable.
    """
    if not answer.strip():
        return GradeResult(
            passed=False,
            score=0.0,
            feedback="No answer submitted.",
            hint="Give it a try -- even a partial answer helps.",
        )

    raw = llm.generate(_build_prompt(question, rubric, evidence, answer, attempt_no))
    try:
        payload = _JudgePayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not parse knowledge grading output: %s", exc)
        return GradeResult(
            passed=False,
            score=0.0,
            feedback="Could not be graded automatically.",
            hint=None,
        )

    return GradeResult(
        passed=payload.passed,
        score=payload.score,
        feedback=payload.feedback,
        hint=None if payload.passed else payload.hint,
    )

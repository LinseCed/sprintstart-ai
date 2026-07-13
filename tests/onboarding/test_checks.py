import json

from llm.base import Message
from onboarding.checks import generate_phase_check
from onboarding.models import PathPhase, PathStep
from rag.types import ScoredChunk
from tests.stubs.llm import StubLLMClient

_STEP = PathStep(id="s1", title="Set up local environment", description="Clone and run")
_CHUNKS: list[ScoredChunk] = []


def _phase(steps: list[PathStep] | None = None) -> PathPhase:
    return PathPhase(title="Setup", steps=steps if steps is not None else [_STEP])


def test_generates_valid_mixed_questions() -> None:
    raw = json.dumps(
        {
            "questions": [
                {
                    "type": "MULTIPLE_CHOICE",
                    "question": "Which command starts the server?",
                    "explanation": "gradlew bootRun starts the backend.",
                    "options": [
                        {"label": "gradlew bootRun", "correct": True},
                        {"label": "npm start", "correct": False},
                    ],
                },
                {
                    "type": "SHORT_TEXT",
                    "question": "What is the start command?",
                    "explanation": "Same command as above.",
                    "correct_answer": "gradlew bootRun",
                },
            ]
        }
    )
    llm = StubLLMClient(generate_response=raw)

    check = generate_phase_check(_phase(), _CHUNKS, llm)

    assert len(check.questions) == 2
    assert check.questions[0].position == 0
    assert check.questions[0].type == "MULTIPLE_CHOICE"
    assert len(check.questions[0].options) == 2
    assert check.questions[0].options[0].correct is True
    assert check.questions[1].position == 1
    assert check.questions[1].type == "SHORT_TEXT"
    assert check.questions[1].correct_answer == "gradlew bootRun"


def test_unparseable_output_degrades_to_empty_check() -> None:
    llm = StubLLMClient(generate_response="not json at all")

    check = generate_phase_check(_phase(), _CHUNKS, llm)

    assert check.questions == []


def test_drops_multiple_choice_question_without_a_correct_option() -> None:
    raw = json.dumps(
        {
            "questions": [
                {
                    "type": "MULTIPLE_CHOICE",
                    "question": "Bad question",
                    "options": [
                        {"label": "A", "correct": False},
                        {"label": "B", "correct": False},
                    ],
                },
                {
                    "type": "SHORT_TEXT",
                    "question": "Good question",
                    "correct_answer": "answer",
                },
            ]
        }
    )
    llm = StubLLMClient(generate_response=raw)

    check = generate_phase_check(_phase(), _CHUNKS, llm)

    assert len(check.questions) == 1
    assert check.questions[0].question == "Good question"
    assert check.questions[0].position == 0


def test_drops_short_text_question_with_blank_correct_answer() -> None:
    raw = json.dumps(
        {
            "questions": [
                {
                    "type": "SHORT_TEXT",
                    "question": "No answer given",
                    "correct_answer": "  ",
                },
            ]
        }
    )
    llm = StubLLMClient(generate_response=raw)

    check = generate_phase_check(_phase(), _CHUNKS, llm)

    assert check.questions == []


def test_phase_without_steps_never_calls_the_llm() -> None:
    class ExplodingLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            raise AssertionError("must not call the LLM for a stepless phase")

    check = generate_phase_check(_phase(steps=[]), _CHUNKS, ExplodingLLM())

    assert check.questions == []

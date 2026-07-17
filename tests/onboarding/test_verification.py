import json

from onboarding.verification import grade_attest, grade_exact, grade_knowledge
from tests.stubs.llm import StubLLMClient


def test_exact_match_passes_on_normalized_equality() -> None:
    result = grade_exact(canonical_answer="  Chroma  DB ", answer="chroma db")

    assert result.passed is True
    assert result.score == 1.0


def test_exact_match_fails_on_mismatch() -> None:
    result = grade_exact(canonical_answer="chroma", answer="pinecone")

    assert result.passed is False
    assert result.hint is not None


def test_exact_match_fails_on_blank_answer() -> None:
    result = grade_exact(canonical_answer="chroma", answer="   ")

    assert result.passed is False


def test_attest_passes_on_nonblank_answer() -> None:
    result = grade_attest(answer="done")

    assert result.passed is True
    assert result.score == 1.0


def test_attest_fails_on_blank_answer() -> None:
    result = grade_attest(answer="")

    assert result.passed is False


def test_knowledge_grading_passes_a_good_answer() -> None:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "passed": True,
                "score": 0.9,
                "feedback": "Correct: covers the key trade-off.",
                "hint": None,
            }
        )
    )

    result = grade_knowledge(
        llm,
        question="Why re-ingest after changing chunking params?",
        rubric="Existing chunks were built with the old params and go stale.",
        evidence="Chunking params affect chunk boundaries.[c1]",
        answer="Because the old chunks no longer match the new chunking logic.",
        attempt_no=1,
    )

    assert result.passed is True
    assert result.score == 0.9
    assert result.hint is None


def test_knowledge_grading_fails_a_vague_answer_with_a_hint() -> None:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "passed": False,
                "score": 0.2,
                "feedback": "Too vague -- doesn't address why re-ingestion matters.",
                "hint": "Think about what happens to already-ingested chunks.",
            }
        )
    )

    result = grade_knowledge(
        llm,
        question="Why re-ingest after changing chunking params?",
        rubric="Existing chunks were built with the old params and go stale.",
        evidence="Chunking params affect chunk boundaries.[c1]",
        answer="idk, seems important",
        attempt_no=1,
    )

    assert result.passed is False
    assert result.hint


def test_knowledge_grading_blank_answer_skips_llm_call() -> None:
    llm = StubLLMClient(generate_response="should never be parsed")

    result = grade_knowledge(
        llm, question="q", rubric="r", evidence="e", answer="   ", attempt_no=1
    )

    assert result.passed is False
    assert result.feedback == "No answer submitted."


def test_knowledge_grading_malformed_output_degrades_to_fail() -> None:
    llm = StubLLMClient(generate_response="not json")

    result = grade_knowledge(
        llm, question="q", rubric="r", evidence="e", answer="an answer", attempt_no=1
    )

    assert result.passed is False
    assert result.feedback == "Could not be graded automatically."


def test_knowledge_grading_never_hints_on_pass() -> None:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "passed": True,
                "score": 1.0,
                "feedback": "Correct.",
                "hint": "a hint the model shouldn't have sent",
            }
        )
    )

    result = grade_knowledge(
        llm,
        question="q",
        rubric="r",
        evidence="e",
        answer="a good answer",
        attempt_no=1,
    )

    assert result.passed is True
    assert result.hint is None

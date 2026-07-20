from onboarding.buddy import build_handoff_prompt
from rag.types import ScoredChunk


def _chunk(filename: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        id=f"c-{filename}",
        artifact_id=f"a-{filename}",
        filename=filename,
        text="...",
        score=score,
    )


def test_handoff_prompt_carries_the_question_and_what_was_checked() -> None:
    messages = build_handoff_prompt(
        "How do I run the migrations?",
        [_chunk("README.md", 0.4), _chunk("db.md", 0.35), _chunk("README.md", 0.32)],
    )

    assert messages[0]["role"] == "system"
    context = messages[-1]["content"]
    # The hire's question is handed to the human verbatim...
    assert "How do I run the migrations?" in context
    # ...along with what was already checked, so they aren't sent to re-read it.
    assert "README.md" in context
    assert "db.md" in context


def test_handoff_prompt_says_nothing_was_found_when_empty() -> None:
    messages = build_handoff_prompt("What's the deploy process?", [])

    context = messages[-1]["content"]
    assert "What's the deploy process?" in context
    assert "Nothing" in context


def test_handoff_prompt_dedups_filenames() -> None:
    messages = build_handoff_prompt("q", [_chunk("a.md", 0.4), _chunk("a.md", 0.3)])

    assert [m["role"] for m in messages] == ["system", "user"]
    # "a.md" appears once in the findings note, not once per chunk.
    assert messages[-1]["content"].count("a.md") == 1

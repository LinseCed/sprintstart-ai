import pytest

from llm.base import Message, ToolSpec
from llm.errors import LLMUnavailableError
from onboarding.buddy_agent import SEARCH_DOCS, run_agent_turn
from rag.types import ScoredChunk
from tests.stubs.llm import ScriptedLLMClient
from tests.stubs.store import StubVectorStore

_GET_MY_METRICS: ToolSpec = {
    "name": "get_my_metrics",
    "description": "The hire's onboarding metrics.",
    "parameters": {"type": "object", "properties": {}},
}


def _user(text: str) -> Message:
    return Message(role="user", content=text)


def _system_of(messages: list[Message]) -> str:
    assert messages[0]["role"] == "system"
    return messages[0].get("content") or ""


class _UnavailableSummarizer(ScriptedLLMClient):
    """Chats fine but cannot summarize -- the degrade path for compaction."""

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        raise LLMUnavailableError("model down")


def test_pauses_and_hands_back_a_backend_tool_call() -> None:
    llm = ScriptedLLMClient(turns=[[("get_my_metrics", {})]])
    store = StubVectorStore()

    result = run_agent_turn([_user("is my PR stuck?")], [_GET_MY_METRICS], llm, store)

    assert result.final is False
    assert [call.name for call in result.pending_tool_calls] == ["get_my_metrics"]
    # The assistant turn carrying the tool call is in the running messages, so the
    # backend can append a matching tool result and resume without it going malformed.
    assert any(
        msg["role"] == "assistant" and msg.get("tool_calls") for msg in result.messages
    )


def test_resumes_with_a_tool_result_and_answers() -> None:
    llm = ScriptedLLMClient(turns=[[("get_my_metrics", {})]])
    store = StubVectorStore()
    first = run_agent_turn([_user("is my PR stuck?")], [_GET_MY_METRICS], llm, store)
    call_id = first.pending_tool_calls[0].id

    # The backend appends the tool result and re-invokes; the scripted model now has
    # no more turns, so it produces its final answer.
    resumed_messages = [
        *first.messages,
        Message(
            role="tool",
            content="openPullRequestCount=1, longestOpenWaitHours=52, stalled=true",
            tool_call_id=call_id,
        ),
    ]
    llm2 = ScriptedLLMClient(
        turns=[], answer="Your PR has waited 52 hours — that's on the reviewer."
    )

    result = run_agent_turn(resumed_messages, [_GET_MY_METRICS], llm2, store)

    assert result.final is True
    assert "52 hours" in result.text


def test_runs_search_docs_locally_and_collects_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = ScoredChunk(
        id="c1",
        artifact_id="a1",
        filename="README.md",
        text="Run ./gradlew build.",
        score=0.9,
    )

    def _fake_retrieve(*args: object, **kwargs: object) -> list[ScoredChunk]:
        return [chunk]

    monkeypatch.setattr("onboarding.buddy_agent.retrieve", _fake_retrieve)
    llm = ScriptedLLMClient(
        turns=[[(SEARCH_DOCS, {"query": "how to build"})], []],
        answer="Run ./gradlew build.",
    )

    result = run_agent_turn(
        [_user("how do I build?")], [_GET_MY_METRICS], llm, StubVectorStore()
    )

    assert result.final is True
    # search_docs is executed here, not handed back, and its chunk becomes a citation.
    assert result.pending_tool_calls == []
    assert [cit.artifact_id for cit in result.citations] == ["a1"]


def test_unknown_tool_is_answered_as_such_and_does_not_stall() -> None:
    llm = ScriptedLLMClient(turns=[[("does_not_exist", {})], []], answer="done")
    result = run_agent_turn([_user("hi")], [_GET_MY_METRICS], llm, StubVectorStore())

    assert result.final is True
    assert any(
        msg["role"] == "tool" and "Unknown tool" in (msg.get("content") or "")
        for msg in result.messages
    )


def test_persona_makes_the_buddy_a_plan_aware_mentor() -> None:
    llm = ScriptedLLMClient(turns=[], answer="hi")

    run_agent_turn([_user("hello")], [_GET_MY_METRICS], llm, StubVectorStore())

    persona = _system_of(llm.chat_calls[0])
    # The load-bearing directives, pinned without over-fitting the wording: plan
    # before recommending, teach from modules, verify through the action, no
    # invented order, no scores.
    assert "get_learning_plan" in persona
    assert "get_module" in persona
    assert "submit_verification" in persona
    assert "claim_goal" in persona
    assert "never invent" in persona
    assert "never mention scores" in persona


def test_folds_the_oldest_messages_into_an_updated_summary() -> None:
    llm = ScriptedLLMClient(
        turns=[], answer="Set up the repo; PR #3 is awaiting review."
    )
    history = [_user(f"m{i}") for i in range(1, 5)]

    result = run_agent_turn(
        history,
        [_GET_MY_METRICS],
        llm,
        StubVectorStore(),
        prior_summary="Earlier notes.",
        summarize_upto=2,
    )

    assert result.updated_summary == "Set up the repo; PR #3 is awaiting review."
    system = _system_of(result.messages)
    # The accreted summary stands in for the folded turns...
    assert "Set up the repo; PR #3 is awaiting review." in system
    # ...and the folded messages are out of the active window, the rest intact.
    contents = [msg.get("content") for msg in result.messages]
    assert "m1" not in contents
    assert "m2" not in contents
    assert "m3" in contents
    assert "m4" in contents


def test_summary_round_trips_inside_the_system_message_on_resume() -> None:
    llm = ScriptedLLMClient(turns=[], answer="Summary of old turns.")
    first = run_agent_turn(
        [_user("m1"), _user("m2")],
        [_GET_MY_METRICS],
        llm,
        StubVectorStore(),
        summarize_upto=1,
    )

    # A resume hop carries the returned conversation verbatim -- no summary fields --
    # and must not get a second persona prepended nor lose the folded memory.
    llm2 = ScriptedLLMClient(turns=[], answer="answer")
    resumed = run_agent_turn(first.messages, [_GET_MY_METRICS], llm2, StubVectorStore())

    system_messages = [m for m in llm2.chat_calls[0] if m["role"] == "system"]
    assert len(system_messages) == 1
    assert "Summary of old turns." in (system_messages[0].get("content") or "")
    assert resumed.updated_summary is None


def test_prior_summary_is_standing_context_without_a_fold_request() -> None:
    llm = ScriptedLLMClient(turns=[], answer="answer")

    result = run_agent_turn(
        [_user("recent")],
        [_GET_MY_METRICS],
        llm,
        StubVectorStore(),
        prior_summary="The hire merged their first PR.",
    )

    assert "The hire merged their first PR." in _system_of(llm.chat_calls[0])
    assert result.updated_summary is None


def test_compaction_degrades_to_no_fold_when_the_model_is_unavailable() -> None:
    llm = _UnavailableSummarizer(turns=[], answer="answer anyway")
    history = [_user(f"m{i}") for i in range(1, 5)]

    result = run_agent_turn(
        history,
        [_GET_MY_METRICS],
        llm,
        StubVectorStore(),
        summarize_upto=2,
    )

    # Nothing is summarized, so nothing is dropped: the whole window stays, and the
    # caller's cursor simply does not advance.
    assert result.updated_summary is None
    contents = [msg.get("content") for msg in result.messages]
    assert all(f"m{i}" in contents for i in range(1, 5))
    assert result.final is True

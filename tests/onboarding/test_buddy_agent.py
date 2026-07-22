import pytest

from llm.base import Message, ToolSpec
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

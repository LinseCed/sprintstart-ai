from pydantic import BaseModel

from agents.base import (
    Agent,
    _merge_into,  # pyright: ignore[reportPrivateUsage]
    _parse_tool_call,  # pyright: ignore[reportPrivateUsage]
)
from agents.tools.agent_tool import AgentTaskArgs, AgentTool
from agents.tools.base import Invocation, Tool, ToolRegistry, ToolResult
from rag.types import ScoredChunk
from tests.stubs.llm import ScriptedLLMClient

_GIVE_CALL = '<tool_call>{"name": "give", "args": {}}</tool_call>'


def _scored(chunk_id: str) -> ScoredChunk:
    return ScoredChunk(
        id=chunk_id, artifact_id="d", filename="f.md", text="hello", score=1.0
    )


class _Args(BaseModel):
    pass


class _ChunkTool(Tool[_Args]):
    name = "give"
    description = "gives a chunk"
    args_schema = "{}"
    args_model = _Args

    def __init__(self) -> None:
        self._calls = 0

    def run(self, args: _Args) -> ToolResult:  # noqa: ARG002
        self._calls += 1
        return ToolResult(summary="gave", chunks=[_scored(f"c{self._calls}")])


def _agent(llm: ScriptedLLMClient, **kwargs: object) -> Agent:
    return Agent(
        name="t",
        description="d",
        llm=llm,
        tools=ToolRegistry([_ChunkTool()]),
        decision_role="r",
        answer_system="s",
        **kwargs,  # type: ignore[arg-type]
    )


def test_parse_tool_call_extracts_valid_call() -> None:
    parsed = _parse_tool_call(_GIVE_CALL, frozenset({"give"}))
    assert parsed == ("give", {})


def test_parse_tool_call_rejects_unknown_name() -> None:
    assert _parse_tool_call(_GIVE_CALL, frozenset({"other"})) is None


def test_parse_tool_call_returns_none_without_call() -> None:
    assert _parse_tool_call("READY", frozenset({"give"})) is None


def test_parse_tool_call_accepts_bare_json_with_noise() -> None:
    raw = 'EXC{"name": "synthesis", "args": {"task": "overview"}}ić'
    assert _parse_tool_call(raw, frozenset({"synthesis"})) == (
        "synthesis",
        {"task": "overview"},
    )


def test_parse_tool_call_accepts_fenced_json() -> None:
    raw = '```json\n{"name": "give", "args": {"x": 1}}\n```'
    assert _parse_tool_call(raw, frozenset({"give"})) == ("give", {"x": 1})


def test_parse_tool_call_ignores_json_without_a_known_tool() -> None:
    assert _parse_tool_call('{"foo": "bar"}', frozenset({"give"})) is None
    assert _parse_tool_call('{"name": "nope"}', frozenset({"give"})) is None


def test_parse_tool_call_prefers_first_valid_object() -> None:
    raw = 'noise {"name": "skip"} then {"name": "give", "args": {}}'
    assert _parse_tool_call(raw, frozenset({"give"})) == ("give", {})


def test_parse_tool_call_accepts_function_call_syntax() -> None:
    raw = 'retrieve({"query": "sprintstart-ai"})'
    assert _parse_tool_call(raw, frozenset({"retrieve", "grep"})) == (
        "retrieve",
        {"query": "sprintstart-ai"},
    )


def test_parse_function_call_ignores_unknown_name() -> None:
    assert _parse_tool_call('search({"q": "x"})', frozenset({"retrieve"})) is None


def test_merge_into_deduplicates_by_id() -> None:
    target = [_scored("a")]
    added = _merge_into(target, [_scored("a"), _scored("b")])
    assert added == 1
    assert [c.id for c in target] == ["a", "b"]


def test_gather_collects_chunks_observations_and_usages() -> None:
    agent = _agent(ScriptedLLMClient([_GIVE_CALL, "READY"]))

    state = agent.gather("question")

    assert [c.id for c in state.chunks] == ["c1"]
    assert state.usages == [Invocation(kind="tool", name="give")]
    assert state.observations == ["[give] gave"]


def test_gather_stops_when_no_tool_call() -> None:
    agent = _agent(ScriptedLLMClient(["just answering directly"]))

    state = agent.gather("question")

    assert state.chunks == []
    assert state.usages == []


def test_gather_respects_step_budget() -> None:
    agent = _agent(
        ScriptedLLMClient([_GIVE_CALL, _GIVE_CALL, _GIVE_CALL]), max_steps=2
    )

    state = agent.gather("question")

    assert [c.id for c in state.chunks] == ["c1", "c2"]


def test_run_returns_answer_and_context() -> None:
    agent = _agent(ScriptedLLMClient([_GIVE_CALL, "READY"], answer="done"))

    result = agent.run("question")

    assert result.answer == "done"
    assert [c.id for c in result.chunks] == ["c1"]
    assert result.usages == [Invocation(kind="tool", name="give")]


def test_answer_stream_yields_tokens() -> None:
    agent = _agent(ScriptedLLMClient([], answer="streamed"))

    tokens = list(agent.answer_stream("q", agent.gather("q")))

    assert "".join(tokens) == "streamed"


def test_agent_tool_delegates_to_subagent() -> None:
    sub = _agent(ScriptedLLMClient([_GIVE_CALL, "READY"], answer="sub answer"))
    sub.name = "rag"
    sub.description = "knowledge base"

    tool = AgentTool(sub)

    assert tool.name == "rag"
    assert tool.kind == "agent"
    assert tool.description == "knowledge base"

    result = tool.run(AgentTaskArgs(task="look it up"))

    assert result.summary == "sub answer"
    assert [c.id for c in result.chunks] == ["c1"]
    # The sub-agent's internal tool use bubbles up through the result.
    assert result.usages == [Invocation(kind="tool", name="give")]


def test_gather_stream_yields_invocations_live() -> None:
    agent = _agent(ScriptedLLMClient([_GIVE_CALL, "READY"]))

    streamed = list(agent.gather_stream("question"))

    assert streamed == [Invocation(kind="tool", name="give")]


def test_gather_stream_forwards_nested_invocations_in_order() -> None:
    sub = _agent(ScriptedLLMClient([_GIVE_CALL, "READY"]))
    sub.name = "rag"
    rag_call = '<tool_call>{"name": "rag", "args": {"task": "x"}}</tool_call>'
    parent = Agent(
        name="orchestrator",
        description="d",
        llm=ScriptedLLMClient([rag_call, "READY"]),
        tools=ToolRegistry([AgentTool(sub)]),
        decision_role="r",
        answer_system="s",
    )

    streamed = list(parent.gather_stream("question"))

    assert streamed == [
        Invocation(kind="agent", name="rag"),
        Invocation(kind="tool", name="give"),
    ]

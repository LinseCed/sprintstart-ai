import json
from collections.abc import Iterator

from agents.orchestrator import ChatOrchestrator
from api.schemas import HistoryEntry
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import ScriptedLLMClient
from tests.stubs.store import StubVectorStore

_RETRIEVE_CALL = ("retrieve", {"query": "blockers"})
_EMBEDDING = [1.0] + [0.0] * 767


def _events(orchestrator: ChatOrchestrator, query: str) -> list[dict[str, object]]:
    history: list[HistoryEntry] = []
    return parse_sse_events("".join(orchestrator.stream(query, history)))


def _store_with_chunk() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="d1",
                filename="retro.md",
                text="missing designs blocked auth",
                embedding=_EMBEDDING,
                heading_path="Blockers",
            )
        ]
    )
    return store


def test_orchestrator_reports_nested_tool_use_in_order() -> None:
    store = _store_with_chunk()
    llm = ScriptedLLMClient(
        [[_RETRIEVE_CALL]],
        answer="Missing designs.",
        embedding=_EMBEDDING,
    )

    events = _events(ChatOrchestrator(llm, store), "What were the blockers?")
    types = [e["type"] for e in events]

    tool_uses = [
        {"name": e["name"], "kind": e["kind"]}
        for e in events
        if e["type"] == "tool_use"
    ]
    assert tool_uses == [
        {"name": "synthesis", "kind": "agent"},
        {"name": "retrieve", "kind": "tool"},
        {"name": "retrieve", "kind": "tool"},
    ]
    assert types.index("tool_use") < types.index("token")

    citations = [e for e in events if e["type"] == "citation"]
    assert citations[0]["filename"] == "retro.md"
    assert events[-1] == {"type": "done"}


def test_orchestrator_streams_single_delegation_without_re_synthesising() -> None:
    store = _store_with_chunk()
    llm = ScriptedLLMClient(
        [[_RETRIEVE_CALL]],
        answer="Missing designs.",
        embedding=_EMBEDDING,
    )

    events = _events(ChatOrchestrator(llm, store), "What were the blockers?")
    tokens = "".join(str(e["content"]) for e in events if e["type"] == "token")

    assert tokens == "Missing designs."
    assert len(llm.stream_calls) == 1


def test_orchestrator_decomposes_compound_query() -> None:
    sub_queries = ["How does authentication work?", "How do I run the tests?"]
    store = _store_with_chunk()
    llm = _PlanningLLM(sub_queries, answer="A grounded answer.", embedding=_EMBEDDING)

    query = (
        "How does authentication work in this project "
        "and how do I run the integration test suite locally?"
    )
    events = _events(ChatOrchestrator(llm, store), query)

    agent_uses = [e for e in events if e["type"] == "tool_use" and e["kind"] == "agent"]
    assert len(agent_uses) == len(sub_queries)

    tokens = "".join(str(e["content"]) for e in events if e["type"] == "token")
    assert tokens.count("## ") == len(sub_queries)
    for sub_query in sub_queries:
        assert f"## {sub_query}" in tokens

    citations = [e for e in events if e["type"] == "citation"]
    assert len(citations) == 1


def test_orchestrator_answers_directly_when_no_delegation() -> None:
    llm = ScriptedLLMClient([], answer="hello there")

    events = _events(ChatOrchestrator(llm, StubVectorStore()), "hi")

    assert not [e for e in events if e["type"] == "tool_use"]
    assert not [e for e in events if e["type"] == "citation"]
    assert events[-1] == {"type": "done"}


def test_orchestrator_emits_error_event_when_llm_unavailable() -> None:
    class _FailingStream(ScriptedLLMClient):
        def stream(self, messages: list[Message]) -> Iterator[str]:
            raise LLMUnavailableError("http://localhost:11434")

    llm = _FailingStream([])

    events = _events(ChatOrchestrator(llm, StubVectorStore()), "hi")

    assert events[0]["type"] == "error"


class _PlanningLLM(ScriptedLLMClient):
    def __init__(
        self, sub_queries: list[str], *, answer: str, embedding: list[float]
    ) -> None:
        super().__init__([], answer=answer, embedding=embedding)
        self._plan = json.dumps({"type": "compound", "sub_queries": sub_queries})

    def generate(self, messages: list[Message]) -> str:
        if "sub_queries" in messages[0]["content"]:
            return self._plan
        return super().generate(messages)

from collections.abc import Iterator

from agents.orchestrator import ChatOrchestrator
from api.schemas import HistoryEntry
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import ScriptedLLMClient
from tests.stubs.store import StubVectorStore

_SYNTH_CALL = ("synthesis", {"task": "blockers"})
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
            )
        ]
    )
    return store


def test_orchestrator_reports_nested_tool_use_in_order() -> None:
    store = _store_with_chunk()
    llm = ScriptedLLMClient(
        [[_SYNTH_CALL], [_RETRIEVE_CALL], [], []],
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
        {"name": "retrieve", "kind": "tool"},  # seed retrieval, before the loop
        {"name": "retrieve", "kind": "tool"},  # the agent's own in-loop retrieve
    ]
    assert types.index("tool_use") < types.index("token")

    citations = [e for e in events if e["type"] == "citation"]
    assert citations[0]["filename"] == "retro.md"
    assert events[-1] == {"type": "done"}


def test_orchestrator_streams_citation_before_answer_tokens() -> None:
    """Citations must be emitted as soon as their chunks are gathered, not
    batched after the answer — a consumer should see the source before (or
    interleaved with) the tokens it grounds, not only once streaming ends."""
    store = _store_with_chunk()
    llm = ScriptedLLMClient(
        [[_SYNTH_CALL], [_RETRIEVE_CALL], [], []],
        answer="Missing designs.",
        embedding=_EMBEDDING,
    )

    events = _events(ChatOrchestrator(llm, store), "What were the blockers?")
    types = [e["type"] for e in events]

    assert "citation" in types
    assert types.index("citation") < types.index("token")

    citations = [e for e in events if e["type"] == "citation"]
    assert len(citations) == 1  # not re-emitted again at the end of the stream


def test_orchestrator_does_not_duplicate_citations_across_multiple_retrieves() -> None:
    """The same chunk id surfacing from more than one tool call (e.g. the
    seed retrieval and the agent's own retrieve both matching) must only
    produce a single citation event, however many times it is re-gathered."""
    store = _store_with_chunk()
    llm = ScriptedLLMClient(
        [[_SYNTH_CALL], [_RETRIEVE_CALL], [_RETRIEVE_CALL], []],
        answer="Missing designs.",
        embedding=_EMBEDDING,
    )

    events = _events(ChatOrchestrator(llm, store), "What were the blockers?")
    citations = [e for e in events if e["type"] == "citation"]

    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "c1"


def test_orchestrator_streams_single_delegation_without_re_synthesising() -> None:
    store = _store_with_chunk()
    llm = ScriptedLLMClient(
        [[_SYNTH_CALL], [_RETRIEVE_CALL], [], []],
        answer="Missing designs.",
        embedding=_EMBEDDING,
    )

    events = _events(ChatOrchestrator(llm, store), "What were the blockers?")
    tokens = "".join(str(e["content"]) for e in events if e["type"] == "token")

    assert tokens == "Missing designs."
    assert len(llm.stream_calls) == 1


def test_orchestrator_chats_directly_without_touching_the_knowledge_base() -> None:
    llm = ScriptedLLMClient([], answer="Hi! How can I help with your project?")

    events = _events(ChatOrchestrator(llm, _store_with_chunk()), "hey there")

    assert not [e for e in events if e["type"] == "tool_use"]
    assert not [e for e in events if e["type"] == "citation"]
    tokens = "".join(str(e["content"]) for e in events if e["type"] == "token")
    assert tokens == "Hi! How can I help with your project?"
    assert events[-1] == {"type": "done"}


def test_orchestrator_emits_error_event_when_llm_unavailable() -> None:
    class _FailingStream(ScriptedLLMClient):
        def stream(self, messages: list[Message]) -> Iterator[str]:
            raise LLMUnavailableError("http://localhost:11434")

    llm = _FailingStream([])

    events = _events(ChatOrchestrator(llm, StubVectorStore()), "hi")

    assert events[0]["type"] == "error"

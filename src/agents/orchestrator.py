import json
import logging
from collections.abc import Generator, Iterator

from agents.base import AgentRunState
from agents.orchestrator_agent import OrchestratorAgent
from agents.tools.base import Invocation
from api.schemas import HistoryEntry
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from store.base import VectorStore

logger = logging.getLogger(__name__)


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _emit_tool_use(
    gen: Generator[Invocation, None, AgentRunState],
) -> Generator[str, None, AgentRunState]:
    while True:
        try:
            usage = next(gen)
        except StopIteration as stop:
            return stop.value
        yield _sse({"type": "tool_use", "name": usage.name, "kind": usage.kind})


class ChatOrchestrator:
    def __init__(self, llm: LLMClient, store: VectorStore) -> None:
        self._agent = OrchestratorAgent(llm, store)

    def stream(self, query: str, history: list[HistoryEntry]) -> Iterator[str]:
        messages: list[Message] = [
            Message(role=h.role, content=h.content) for h in history
        ]

        try:
            state = yield from _emit_tool_use(
                self._agent.gather_stream(query, messages)
            )

            for token in self._agent.answer_stream(query, state, messages):
                if token:
                    yield _sse({"type": "token", "content": token})

            for citation in build_citations(state.chunks):
                yield _sse(
                    {
                        "type": "citation",
                        "chunk_id": citation.chunk_id,
                        "filename": citation.filename,
                        "section_path": citation.section_path,
                    }
                )

            yield _sse({"type": "done"})

        except LLMUnavailableError as exc:
            yield _sse({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in chat stream")
            yield _sse({"type": "error", "message": "An unexpected error occurred"})

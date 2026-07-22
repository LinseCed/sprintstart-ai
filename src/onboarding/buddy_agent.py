"""Agentic onboarding buddy: one tool-using turn.

The buddy is no longer a corpus-only Q&A box. It reasons over the hire's question
and calls tools -- some it runs itself, some only the backend can. ``search_docs``
is AI-local (it owns retrieval and citations) and is executed here, in an internal
loop, so a question needing several searches is answered in one call. A tool only
the backend can run (``get_my_metrics``, "is my PR stuck?" -- backend-owned data)
can't be executed here: the turn stops and hands the pending call back, and the
backend re-invokes this endpoint with the tool result appended.

Stateless like every other onboarding endpoint: the caller (backend) carries the
running message list between invocations. Nothing about the hire lives here -- their
state arrives only as tool results the backend supplies.
"""

from dataclasses import dataclass, field

from llm.base import ChatResult, LLMClient, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from rag.retriever import retrieve
from rag.source_filter import SourceExclusions
from rag.types import Citation, ScoredChunk
from store.base import VectorStore

_PERSONA = (
    "You are the onboarding buddy: the mentor a new hire would otherwise have to "
    "find a person for. You are warm, patient, and always available -- no question "
    "is too basic. You have tools: use `search_docs` for anything about how this "
    "codebase, product, or process works, and use the hire-state tools (e.g. "
    "`get_my_metrics`) for questions about the hire's own onboarding -- their pull "
    "requests, whether they're stuck, how long a review has been waiting. Prefer a "
    "grounded answer from a tool over guessing. When a tool gives you the facts, "
    "answer plainly and encouragingly; if the tools don't cover it, say so honestly "
    "rather than inventing an answer."
)

SEARCH_DOCS = "search_docs"

_SEARCH_TOOL: ToolSpec = {
    "name": SEARCH_DOCS,
    "description": (
        "Search the project's indexed documentation, code, issues and pull requests "
        "for grounded evidence. Use this for any question about how the codebase, "
        "product, or process works."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, phrased as a search query.",
            }
        },
        "required": ["query"],
    },
}

_TOP_K = 5
# Same retrieval floor / confidence line as the legacy buddy: `retrieve` drops
# anything below it, so an empty result means nothing indexed answers with confidence.
_MIN_SCORE = 0.3
# How many internal search hops before we force a final answer, so a confused model
# can't loop forever gathering evidence it never uses.
_MAX_STEPS = 4


@dataclass
class AgentTurnResult:
    """The outcome of one agent turn.

    ``final`` distinguishes "here is the answer" (``text`` is it) from "I need the
    backend to run these tools first" (``pending_tool_calls`` are them). ``messages``
    is always the full running conversation the caller must carry back verbatim next
    turn -- it already includes any search steps run here and the tool-use turn the
    pending calls belong to.
    """

    final: bool
    text: str
    messages: list[Message]
    pending_tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    citations: list[Citation] = field(default_factory=list[Citation])


def _ensure_persona(messages: list[Message]) -> list[Message]:
    if messages and messages[0]["role"] == "system":
        return list(messages)
    return [Message(role="system", content=_PERSONA), *messages]


def _assistant_message(result: ChatResult) -> Message:
    msg = Message(role="assistant", content=result.text)
    if result.tool_calls:
        msg["tool_calls"] = result.tool_calls
    return msg


def _tool_result_message(call_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id)


def _format_chunks(chunks: list[ScoredChunk]) -> str:
    if not chunks:
        return "No indexed material matched this search."
    parts = [f"[{chunk.filename}]\n{chunk.text}" for chunk in chunks]
    return "\n\n---\n\n".join(parts)


def run_agent_turn(
    messages: list[Message],
    backend_tools: list[ToolSpec],
    llm: LLMClient,
    store: VectorStore,
    exclusions: SourceExclusions | None = None,
) -> AgentTurnResult:
    """Runs one agent turn: executes ``search_docs`` locally, pauses on backend tools.

    Loops internally while the model only asks for local searches; returns as soon as
    it either produces a final answer or requests a tool only the backend can run. A
    step budget bounds the internal loop; if it's exhausted the model is asked once
    more with no tools, forcing an answer.
    """
    work = _ensure_persona(messages)
    tools = [_SEARCH_TOOL, *backend_tools]
    backend_names = {tool["name"] for tool in backend_tools}
    resolved_exclusions = exclusions if exclusions is not None else SourceExclusions()
    citations: list[Citation] = []

    for _ in range(_MAX_STEPS):
        result = llm.chat(work, tools)
        work = [*work, _assistant_message(result)]

        if not result.tool_calls:
            return AgentTurnResult(
                final=True, text=result.text, messages=work, citations=citations
            )

        pending: list[ToolCall] = []
        for call in result.tool_calls:
            if call.name == SEARCH_DOCS:
                query = str(call.arguments.get("query", "")).strip()
                chunks = retrieve(
                    query,
                    llm,
                    store,
                    top_k=_TOP_K,
                    min_score=_MIN_SCORE,
                    exclusions=resolved_exclusions,
                )
                citations.extend(build_citations(chunks))
                work = [*work, _tool_result_message(call.id, _format_chunks(chunks))]
            elif call.name in backend_names:
                pending.append(call)
            else:
                work = [
                    *work,
                    _tool_result_message(call.id, f"Unknown tool: {call.name}."),
                ]

        # A tool only the backend can run: stop and hand it back. Any local searches
        # in this same turn already have their results appended above, so the message
        # list stays well-formed.
        if pending:
            return AgentTurnResult(
                final=False,
                text=result.text,
                messages=work,
                pending_tool_calls=pending,
                citations=citations,
            )
        # Only local searches this turn -- loop and let the model reason over them.

    # Step budget spent: force a final answer with no tools rather than loop forever.
    forced = llm.generate(work)
    return AgentTurnResult(
        final=True,
        text=forced,
        messages=[*work, Message(role="assistant", content=forced)],
        citations=citations,
    )


__all__ = ["AgentTurnResult", "LLMUnavailableError", "run_agent_turn", "SEARCH_DOCS"]

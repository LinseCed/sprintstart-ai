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

Session memory: the backend bounds an unbounded transcript by sending only a recent
window plus a running summary of everything older (``prior_summary``). When it asks
(``summarize_upto``), this side folds the oldest window messages into the summary --
a language task, so it lives here; persistence is the backend's. The summary rides
the system message in the returned conversation, so resume hops need nothing re-sent.
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
    "You are the onboarding buddy: the mentor who guides a new hire from their first "
    "day to shipping real work. You are warm, patient, and always available -- no "
    "question is too basic.\n"
    "How you work:\n"
    "- You have a plan for the hire, not just answers. Before recommending what to "
    "learn or work on, consult `get_learning_plan`: it says what they are working "
    "toward, which competencies come next in the order the team's graph suggests, "
    "and why. The plan determines sequence -- never invent your own curriculum "
    "order. Relay its reasons ('usually comes after X') as guidance, never as "
    "gates, and never mention scores.\n"
    "- Teach from the team's shared material: when the hire should learn a "
    "competency, read its module with `get_module` and teach from its pages, citing "
    "the sources they carry. If no module exists, teach from the docs with "
    "`search_docs` instead -- and never fabricate module content.\n"
    "- Use `search_docs` for anything about how this codebase, product, or process "
    "works, and the hire-state tools (`get_my_metrics`, `get_my_competencies`, "
    "`get_suggested_tasks`) for the hire's own progress.\n"
    "- Ground every claim about the codebase in a tool result -- a module page's "
    "sources or `search_docs` citations. If the tools don't cover it, say so "
    "honestly rather than inventing an answer; offer `flag_to_pm` as the last "
    "resort.\n"
    "- Turn work into record: when the hire has done what a module's check asks, "
    "offer to submit their answer with `submit_verification` -- you relay the "
    "verdict, you never grade the work yourself. When they pick a suggested task, "
    "offer `claim_goal`.\n"
    "- Celebrate the merges and milestones the metrics report. Shipping is the "
    "point; everything else is the path to it."
)

# How the summary enters the model's context: appended to the persona in the system
# message, so it rides the running conversation the caller carries between turns.
_SUMMARY_HEADER = "\n\nConversation so far (compressed memory of earlier turns):\n"

_COMPACT_SYSTEM = (
    "You compress a running mentorship conversation into a durable memory note for "
    "the mentor's future self. Third person, factual, under 200 words. Keep: what "
    "the hire is working toward, tasks claimed or merged, pull requests opened, "
    "what they have been taught, decisions made, open questions. Drop: greetings, "
    "phatic talk, superseded questions, anything the recent window still covers."
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
    pending calls belong to. ``updated_summary`` is set only when the caller asked
    for compaction (``summarize_upto``) and the fold succeeded; the backend persists
    it and advances its cursor.
    """

    final: bool
    text: str
    messages: list[Message]
    pending_tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    citations: list[Citation] = field(default_factory=list[Citation])
    updated_summary: str | None = None


def _persona_prompt(summary: str | None) -> str:
    if not summary:
        return _PERSONA
    return _PERSONA + _SUMMARY_HEADER + summary


def _ensure_persona(messages: list[Message], summary: str | None) -> list[Message]:
    # A system message already leads the running conversation on a resume hop: the
    # summary is folded inside it, so nothing is re-sent or double-folded.
    if messages and messages[0]["role"] == "system":
        return list(messages)
    return [Message(role="system", content=_persona_prompt(summary)), *messages]


def _try_compact(
    prior_summary: str | None,
    folded: list[Message],
    llm: LLMClient,
) -> str | None:
    """Folds [folded] into the running summary, or None when compaction can't run.

    An unavailable model degrades to no compaction (the caller keeps the whole
    window and its cursor) rather than failing the turn -- the summary is a
    prompt-shaping device, never something a hire's question should 503 over.
    Deterministic (temperature 0) so the same conversation compacts the same way.
    """
    transcript = "\n".join(
        f"{msg['role']}: {msg.get('content') or ''}"
        for msg in folded
        if msg.get("content")
    )
    if not transcript.strip():
        return prior_summary or ""
    prompt = [
        Message(role="system", content=_COMPACT_SYSTEM),
        Message(
            role="user",
            content=(
                f"Memory so far:\n{prior_summary or '(nothing yet)'}\n\n"
                "Conversation turns sliding out of the active window:\n"
                f"{transcript}\n\n"
                "Update the memory note."
            ),
        ),
    ]
    try:
        return llm.generate(prompt, temperature=0)
    except LLMUnavailableError:
        return None


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
    prior_summary: str | None = None,
    summarize_upto: int | None = None,
) -> AgentTurnResult:
    """Runs one agent turn: executes ``search_docs`` locally, pauses on backend tools.

    Loops internally while the model only asks for local searches; returns as soon as
    it either produces a final answer or requests a tool only the backend can run. A
    step budget bounds the internal loop; if it's exhausted the model is asked once
    more with no tools, forcing an answer.

    ``prior_summary`` stands in for everything older than ``messages``; when
    ``summarize_upto`` is set, that many leading messages are folded into the summary
    first (``updated_summary`` on the result) and drop out of the active window.
    """
    updated_summary: str | None = None
    window = list(messages)
    summary = prior_summary
    if summarize_upto and summarize_upto > 0:
        folded = _try_compact(prior_summary, window[:summarize_upto], llm)
        # Only a successful fold advances anything: a failed one keeps the whole
        # window, so nothing the model has not summarized is ever dropped.
        if folded is not None:
            updated_summary = folded
            summary = folded
            window = window[summarize_upto:]

    work = _ensure_persona(window, summary)
    tools = [_SEARCH_TOOL, *backend_tools]
    backend_names = {tool["name"] for tool in backend_tools}
    resolved_exclusions = exclusions if exclusions is not None else SourceExclusions()
    citations: list[Citation] = []

    for _ in range(_MAX_STEPS):
        result = llm.chat(work, tools)
        work = [*work, _assistant_message(result)]

        if not result.tool_calls:
            return AgentTurnResult(
                final=True,
                text=result.text,
                messages=work,
                citations=citations,
                updated_summary=updated_summary,
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
                updated_summary=updated_summary,
            )
        # Only local searches this turn -- loop and let the model reason over them.

    # Step budget spent: force a final answer with no tools rather than loop forever.
    forced = llm.generate(work)
    return AgentTurnResult(
        final=True,
        text=forced,
        messages=[*work, Message(role="assistant", content=forced)],
        citations=citations,
        updated_summary=updated_summary,
    )


__all__ = ["AgentTurnResult", "LLMUnavailableError", "run_agent_turn", "SEARCH_DOCS"]

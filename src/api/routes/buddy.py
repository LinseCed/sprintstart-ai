import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_source_state_store, get_store
from api.schemas import (
    BuddyAgentMessageSchema,
    BuddyAgentRequest,
    BuddyAgentResponse,
    BuddyCitationSchema,
    BuddyToolCallSchema,
    BuddyToolSpecSchema,
    ChatRequest,
    ValidationErrorResponse,
)
from api.sse import sse_event
from ingestion.source_state_store import SourceStateStore
from llm.base import LLMClient, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from onboarding.buddy import build_buddy_prompt, build_handoff_prompt
from onboarding.buddy_agent import run_agent_turn
from rag.citation import build_citations
from rag.retriever import retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

_TOP_K = 5
# The retrieval floor is also the confidence line: `retrieve` drops anything below
# it, so an empty result means nothing indexed answers this with confidence. And if
# the answer *were* in the docs, the buddy would have grounded it here -- so a
# hand-off inherently means "not in the indexed material", not "not looked for".
_MIN_SCORE = 0.3


@router.post(
    "/onboarding/buddy",
    summary="Ask the persistent onboarding buddy a question (streaming)",
    response_class=StreamingResponse,
    tags=["onboarding-buddy"],
    responses={422: {"model": ValidationErrorResponse}},
)
def buddy(
    body: ChatRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    source_state: SourceStateStore = Depends(get_source_state_store),
) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        try:
            yield sse_event({"type": "tool_use", "name": "retrieve", "kind": "tool"})

            chunks = retrieve(
                body.question,
                llm,
                store,
                top_k=_TOP_K,
                min_score=_MIN_SCORE,
                exclusions=source_state.get_exclusions(),
            )

            # Nothing indexed answers this with confidence: don't guess -- hand
            # off and draft the question the hire should ask a person instead.
            if not chunks:
                yield from _draft_handoff(body.question, chunks, llm)
                return

            history: list[Message] = [
                Message(role=h.role, content=h.content) for h in body.history
            ]
            messages = build_buddy_prompt(body.question, chunks, history)

            for token in llm.stream(messages):
                if token:
                    yield sse_event({"type": "token", "content": token})

            for citation in build_citations(chunks):
                yield sse_event(
                    {
                        "type": "citation",
                        "artifact_id": citation.artifact_id,
                        "start_line": citation.start_line,
                        "start_page": citation.start_page,
                    }
                )

            yield sse_event({"type": "done"})

        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in buddy stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _to_message(schema: BuddyAgentMessageSchema) -> Message:
    msg = Message(role=schema.role, content=schema.content)
    if schema.tool_calls:
        msg["tool_calls"] = [
            ToolCall(id=call.id, name=call.name, arguments=dict(call.arguments))
            for call in schema.tool_calls
        ]
    if schema.tool_call_id is not None:
        msg["tool_call_id"] = schema.tool_call_id
    return msg


def _from_message(msg: Message) -> BuddyAgentMessageSchema:
    return BuddyAgentMessageSchema(
        role=msg["role"],
        content=msg.get("content") or "",
        tool_calls=[
            BuddyToolCallSchema(
                id=call.id, name=call.name, arguments=dict(call.arguments)
            )
            for call in msg.get("tool_calls") or []
        ],
        tool_call_id=msg.get("tool_call_id"),
    )


def _to_toolspec(schema: BuddyToolSpecSchema) -> ToolSpec:
    return ToolSpec(
        name=schema.name,
        description=schema.description,
        parameters=dict(schema.parameters),
    )


@router.post(
    "/onboarding/buddy/agent",
    response_model=BuddyAgentResponse,
    summary="Run one agentic buddy turn (tool-using, stateless)",
    tags=["onboarding-buddy"],
    responses={422: {"model": ValidationErrorResponse}},
)
def buddy_agent(
    body: BuddyAgentRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    source_state: SourceStateStore = Depends(get_source_state_store),
) -> BuddyAgentResponse:
    """One turn of the tool-using buddy.

    Executes ``search_docs`` locally (retrieval + citations) and returns as soon as it
    either has a final answer or needs a backend-only tool run. The backend carries the
    ``messages`` list back verbatim, each pending tool's result appended as a ``tool``.
    """
    messages = [_to_message(m) for m in body.messages]
    backend_tools = [_to_toolspec(t) for t in body.backend_tools]
    try:
        result = run_agent_turn(
            messages,
            backend_tools,
            llm,
            store,
            exclusions=source_state.get_exclusions(),
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return BuddyAgentResponse(
        final=result.final,
        text=result.text,
        messages=[_from_message(m) for m in result.messages],
        pending_tool_calls=[
            BuddyToolCallSchema(
                id=call.id, name=call.name, arguments=dict(call.arguments)
            )
            for call in result.pending_tool_calls
        ],
        citations=[
            BuddyCitationSchema(
                artifact_id=cit.artifact_id,
                start_line=cit.start_line,
                start_page=cit.start_page,
            )
            for cit in result.citations
        ],
    )


def _draft_handoff(
    question: str,
    chunks: list[ScoredChunk],
    llm: LLMClient,
) -> Iterator[str]:
    """Streams the hand-off: instead of an answer, the question the hire should
    put to their human buddy, composed from what was (and wasn't) found.

    Emits a ``tool_use`` marker (``name="draft_question"``) before the text so a
    client can tell "answered from the corpus" from "handed to a person" without
    parsing prose; the drafted text itself also says which mode it is in. No
    citations follow -- nothing here is a grounded answer to cite.
    """
    yield sse_event({"type": "tool_use", "name": "draft_question", "kind": "handoff"})

    messages = build_handoff_prompt(question, chunks)
    for token in llm.stream(messages):
        if token:
            yield sse_event({"type": "token", "content": token})

    yield sse_event({"type": "done"})

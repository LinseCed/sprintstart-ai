from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_source_state_store, get_store
from api.schemas import (
    BuddyAgentMessageSchema,
    BuddyAgentRequest,
    BuddyAgentResponse,
    BuddyCitationSchema,
    BuddyOpenRequest,
    BuddyToolCallSchema,
    BuddyToolSpecSchema,
    ValidationErrorResponse,
)
from api.sse import sse_event
from ingestion.source_state_store import SourceStateStore
from llm.base import LLMClient, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from onboarding.buddy_agent import run_agent_turn
from onboarding.buddy_open import stream_session
from onboarding.vocabulary import Vocabulary
from store.base import VectorStore

router = APIRouter()


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
            prior_summary=body.prior_summary,
            summarize_upto=body.summarize_upto,
            vocabulary=Vocabulary(
                contribution_noun=body.vocabulary.contribution_noun,
                contribution_noun_plural=body.vocabulary.contribution_noun_plural,
                contribution_verb_past=body.vocabulary.contribution_verb_past,
            ),
            project_ids=frozenset(body.project_ids) or None,
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
        updated_summary=result.updated_summary,
    )


@router.post(
    "/onboarding/buddy/open/stream",
    summary="Open a buddy visit: refresh the mentor's memory and greet the hire",
    response_class=StreamingResponse,
    tags=["onboarding-buddy"],
    responses={422: {"model": ValidationErrorResponse}},
)
def buddy_open_stream(
    body: BuddyOpenRequest,
    llm: LLMClient = Depends(get_llm),
) -> StreamingResponse:
    """Fold the previous visit into the mentor's memory and greet the hire, streaming.

    ⚠️ **There was a non-streaming version and its ordering was the whole problem.** It
    asked for strict JSON whose first field is the memory note the hire never sees, so
    opening a visit meant waiting for up to 200 words of invisible output before the
    first word addressed to the hire was generated. This one puts the greeting first
    and streams it, for the same single model call and the same tokens.

    Emits ``token`` events carrying the greeting as it arrives and one terminal
    ``done`` carrying the whole greeting, the folded memory and any suggested action —
    the caller persists those exactly as it does for the non-streaming call. Degrades
    to a plain welcome rather than erroring: opening the buddy must never fail the page.
    """

    def event_stream() -> Iterator[str]:
        for event in stream_session(
            memory=body.memory,
            recent=[_to_message(m) for m in body.recent],
            state=body.state,
            llm=llm,
        ):
            yield sse_event(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

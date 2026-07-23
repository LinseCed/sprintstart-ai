from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_llm, get_source_state_store, get_store
from api.schemas import (
    BuddyAgentMessageSchema,
    BuddyAgentRequest,
    BuddyAgentResponse,
    BuddyCitationSchema,
    BuddyOpenActionSchema,
    BuddyOpenRequest,
    BuddyOpenResponse,
    BuddyToolCallSchema,
    BuddyToolSpecSchema,
    ValidationErrorResponse,
)
from ingestion.source_state_store import SourceStateStore
from llm.base import LLMClient, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from onboarding.buddy_agent import run_agent_turn
from onboarding.buddy_open import open_session
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
    "/onboarding/buddy/open",
    response_model=BuddyOpenResponse,
    summary="Open a buddy visit: refresh the mentor's memory and greet the hire",
    tags=["onboarding-buddy"],
    responses={422: {"model": ValidationErrorResponse}},
)
def buddy_open(
    body: BuddyOpenRequest,
    llm: LLMClient = Depends(get_llm),
) -> BuddyOpenResponse:
    """Fold the previous visit into the mentor's memory and write a proactive greeting.

    Degrades to the prior memory and a plain welcome rather than erroring — opening
    the buddy must never fail the page.
    """
    opening = open_session(
        memory=body.memory,
        recent=[_to_message(m) for m in body.recent],
        state=body.state,
        llm=llm,
    )
    action = (
        BuddyOpenActionSchema(label=opening.action_label, question=opening.action_question)
        if opening.action_label and opening.action_question
        else None
    )
    return BuddyOpenResponse(
        memory=opening.memory, greeting=opening.greeting, action=action
    )

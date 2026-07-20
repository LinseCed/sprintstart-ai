import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_source_state_store, get_store
from api.schemas import ChatRequest, ValidationErrorResponse
from api.sse import sse_event
from ingestion.source_state_store import SourceStateStore
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from onboarding.buddy import build_buddy_prompt, build_handoff_prompt
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

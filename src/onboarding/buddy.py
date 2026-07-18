"""Persistent onboarding buddy: a warmly-framed, repo-grounded Q&A companion.

Reuses the same retrieval layer and prompt-building block as ``/chat``
(:func:`rag.retriever.retrieve`, :func:`rag.prompt.build_messages`) but with a
distinct system framing aimed at onboarding's "fear of asking" bottleneck --
unlike ``/chat``'s general-purpose, unopinionated tone, the buddy explicitly
signals that no question is too basic. Stateless like every other onboarding
endpoint: the backend owns conversation history and passes the full
transcript back on every call.
"""

from llm.base import Message
from rag.prompt import build_messages
from rag.types import ScoredChunk

_PERSONA = (
    "You are the onboarding buddy: a warm, patient, always-available companion "
    "for a new hire ramping up on this codebase. Your whole purpose is to make "
    "asking questions feel safe -- no question is too basic, too obvious, or a "
    "waste of anyone's time. If the evidence below answers the question, ground "
    "your answer in it and say so plainly. If it doesn't, say you don't have "
    "grounded evidence for that rather than guessing, and suggest who or where "
    "they might ask next. Keep answers conversational and encouraging, not "
    "terse or formal."
)


def build_buddy_prompt(
    question: str,
    chunks: list[ScoredChunk],
    history: list[Message],
) -> list[Message]:
    """Builds the buddy's message list: persona framing, then the same
    grounded-context + history + question shape ``build_messages`` produces.
    """
    return [
        Message(role="system", content=_PERSONA),
        *build_messages(question, chunks, history),
    ]

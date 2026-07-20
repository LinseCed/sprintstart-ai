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


_HANDOFF_PERSONA = (
    "You are the onboarding buddy, and you could NOT find a confident answer to "
    "the new hire's question in the project's indexed material. Do not guess, "
    "invent an answer, or hedge with a vague one -- the evidence isn't there. "
    "Your job now is the thing a new hire finds hardest: asking a person well. "
    "Write a short, warm hand-off that does three things, in order:\n"
    "1. Say plainly, in one line, that you couldn't find this in the project's "
    "indexed docs -- so it's clear you're handing off, not answering.\n"
    "2. Give them a specific, well-formed question they can paste to their human "
    "buddy: what they're trying to do, and the concrete thing they're stuck on.\n"
    "3. Fold in what was already checked (from 'What the search found' below) so "
    "the human isn't asked something already written down -- e.g. 'I've already "
    "looked at the README and X.md'. If nothing relevant was found, say the "
    "search turned up nothing on this yet.\n"
    "Keep it to a few sentences. Encourage them -- asking is normal and welcome. "
    "Output only the message to show the hire, nothing else."
)


def _findings_summary(chunks: list[ScoredChunk]) -> str:
    """A short, human-readable note of what retrieval turned up, so the drafted
    question can tell the buddy what has already been checked. Empty chunks read
    as 'nothing', weak ones list the files that came closest.
    """
    if not chunks:
        return "Nothing in the indexed material matched this question."
    # De-duplicate filenames while preserving retrieval order (best first).
    seen: list[str] = []
    for chunk in chunks:
        if chunk.filename not in seen:
            seen.append(chunk.filename)
    files = ", ".join(seen[:5])
    return (
        "The closest indexed files were: "
        f"{files} -- but none confidently answered the question."
    )


def build_handoff_prompt(question: str, chunks: list[ScoredChunk]) -> list[Message]:
    """Builds the message list for the hand-off path: rather than answering,
    the buddy drafts the question the hire should put to their human buddy.

    Passes the weak/absent retrieval result as context so the draft can state
    what was already searched -- the point being not to send a human to look up
    something the README already covers.
    """
    context = (
        f"The new hire asked:\n{question}\n\n"
        f"What the search found:\n{_findings_summary(chunks)}"
    )
    return [
        Message(role="system", content=_HANDOFF_PERSONA),
        Message(role="user", content=context),
    ]

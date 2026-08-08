"""Fold the oldest turns of a mentorship conversation into a durable memory note.

A language task, so it lives here; persistence and the cursor are the backend's.

⚠️ **This is a call the hire is not waiting on, and that is the point of it having
its own endpoint.** It used to run only as a step inside
``buddy_agent.run_agent_turn`` -- with its own prompt and its own ``llm.generate``,
so the *quality* was never the problem, but it ran **before** the agent loop. Once a
visit's active window outgrew the backend's ``WINDOW``, every further turn paid an
extra serialized model call before the hire's answer even started being generated,
to fold a single exchange. Exposed on its own, the backend runs it after a turn
finishes instead of in front of one.

The buddy's *open* had the mirror-image defect: there the memory note really did
share a call with the greeting, so the hire's durable memory was composed while the
model was busy greeting them. Both paths now delegate here.
"""

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError

_COMPACT_SYSTEM = (
    "You compress a running mentorship conversation into a durable memory note for "
    "the mentor's future self. Third person, factual, under 200 words. Keep: what "
    # Deliberately role-neutral rather than templated per track: this note is read
    # back by the mentor, never shown to the hire, so neutral wording covers every
    # role without threading a second vocabulary seam through compaction.
    "the hire is working toward, tasks claimed or completed, work they submitted "
    "and what came of it, what they have been taught, decisions made, open "
    "questions. Drop: greetings, "
    "phatic talk, superseded questions, anything the recent window still covers."
)


def compact_memory(
    prior_summary: str | None,
    folded: list[Message],
    llm: LLMClient,
) -> str | None:
    """Folds [folded] into the running memory note, or None when it can't run.

    An unavailable model returns None rather than raising: the caller keeps its
    window and its cursor and tries again later. The note is a prompt-shaping
    device, never the record -- the transcript it compresses stays durable on the
    backend, so a fold that never happens costs a longer prompt and nothing else.

    Deterministic (temperature 0) so the same conversation compacts the same way.

    @param prior_summary: The note as it stands, or None before the first fold.
    @param folded: The messages sliding out of the active window, oldest first.
    @return: The rewritten note, or None when the model was unavailable.
    """
    # ⚠️ Filtered on the *stripped* content, not on the raw field. The version this
    # was extracted from tested `msg.get("content")` for truthiness, so a
    # whitespace-only message survived the filter and contributed its role label --
    # leaving a transcript of `"assistant:"` that the emptiness guard below then read
    # as words. The guard existed and did not hold.
    transcript = "\n".join(
        f"{msg['role']}: {content}"
        for msg in folded
        if (content := (msg.get("content") or "").strip())
    )
    # Nothing with words in it: the note stands rather than being rewritten from
    # nothing. Not None -- this is a successful fold of an empty slice, and the
    # caller may advance its cursor past messages that carried no content.
    if not transcript:
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


__all__ = ["compact_memory"]

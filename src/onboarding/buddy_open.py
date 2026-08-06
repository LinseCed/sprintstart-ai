"""Open a buddy visit: refresh the mentor's memory, then greet the hire.

Unlike replaying a transcript, a visit opens like walking up to a mentor: the
buddy recalls what it knows about the hire (its durable memory), folds in
whatever has happened since it last updated that memory, notes the hire's current
state, and opens with a warm, specific greeting — proactively surfacing the one
thing worth saying rather than waiting to be asked.

Stateless like every onboarding endpoint: the backend supplies the prior memory,
the recent (not-yet-remembered) messages, and a snapshot of the hire's state, and
persists the memory and greeting this returns.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError

_SYSTEM = (
    "You are a warm, perceptive onboarding mentor greeting a new hire as they open "
    "the chat. You keep a private, durable memory note about this hire, and you "
    "speak to them directly.\n"
    "You are given: your MEMORY of the hire (may be empty on the first visit), the "
    "RECENT conversation since you last updated that memory (may be empty), and the "
    "hire's current STATE (their work in flight, tasks, competencies). What the "
    "state contains depends on the hire's role -- describe only what is actually "
    "there, and never assume they write code.\n"
    "Return STRICT JSON with three fields:\n"
    '1. "memory": rewrite your memory note, folding in anything new worth '
    "remembering from the recent conversation — what the hire is working toward, "
    "what you have taught or explained, decisions made, open threads, their "
    "preferences, and what they have struggled with. Third person, factual, concise "
    "(under 200 words). Drop greetings and small talk. If nothing is new, return the "
    "memory unchanged.\n"
    '2. "greeting": a short, warm, first-person opener (2-4 sentences) that greets '
    "the hire and proactively says the one thing most worth saying right now — "
    "grounded in the memory and the current state (work waiting on somebody else, "
    "something of theirs that landed and is worth celebrating, a stall, an open "
    "thread from last time). Be specific, not generic. Never invent facts that are "
    "not in the memory or the state.\n"
    '3. "action": optionally ONE suggested next step, as {"label": short button '
    'text, "question": the message to send when the hire clicks it}, or null when '
    "none fits.\n"
    'Return ONLY the JSON object, nothing else: {"memory": "...", "greeting": '
    '"...", "action": {"label": "...", "question": "..."} | null}.'
)

_FALLBACK_GREETING = "Welcome back! How can I help with your onboarding today?"


@dataclass
class BuddyOpening:
    """The result of opening a visit: the refreshed memory and the greeting to show."""

    memory: str
    greeting: str
    action_label: str | None = None
    action_question: str | None = None


def _format_recent(recent: list[Message]) -> str:
    lines = [
        f"{m['role']}: {m.get('content') or ''}" for m in recent if m.get("content")
    ]
    return "\n".join(lines) if lines else "(nothing since the last memory update)"


def open_session(
    memory: str | None,
    recent: list[Message],
    state: str,
    llm: LLMClient,
) -> BuddyOpening:
    """Fold ``recent`` into ``memory`` and write a greeting grounded in ``state``.

    Degrades to the prior memory and a plain welcome when the model is unavailable
    or returns unparseable output — opening a visit must never fail the page.
    """
    prompt = [
        Message(role="system", content=_SYSTEM),
        Message(
            role="user",
            content=(
                f"MEMORY:\n{memory or '(no memory yet -- first visit)'}\n\n"
                "RECENT conversation since the last memory update:\n"
                f"{_format_recent(recent)}\n\n"
                f"STATE (current):\n{state or '(no state available)'}\n\n"
                "Return the JSON."
            ),
        ),
    ]
    try:
        raw = llm.generate(prompt, temperature=0.3)
    except LLMUnavailableError:
        return BuddyOpening(memory=memory or "", greeting=_FALLBACK_GREETING)
    return _parse(raw, fallback_memory=memory or "")


_MEMORY_MARKER = "<<<MEMORY>>>"
_ACTION_MARKER = "<<<ACTION>>>"

_STREAM_SYSTEM = (
    "You are a warm, perceptive onboarding mentor greeting a new hire as they open "
    "the chat. You keep a private, durable memory note about this hire, and you "
    "speak to them directly.\n"
    "You are given: your MEMORY of the hire (may be empty on the first visit), the "
    "RECENT conversation since you last updated that memory (may be empty), and the "
    "hire's current STATE (their work in flight, tasks, competencies). What the "
    "state contains depends on the hire's role -- describe only what is actually "
    "there, and never assume they write code.\n"
    "Write your reply in exactly three parts, in this order, with nothing before the "
    "first part:\n"
    "PART 1 -- the greeting, as plain prose with no label and no quotes: a short, "
    "warm, first-person opener (2-4 sentences) that greets the hire and proactively "
    "says the one thing most worth saying right now, grounded in the memory and the "
    "current state (work waiting on somebody else, something of theirs that landed "
    "and is worth celebrating, a stall, an open thread from last time). Be specific, "
    "not generic. Never invent facts that are not in the memory or the state. The "
    "hire reads this as you type it, so it must come first.\n"
    f"PART 2 -- the line {_MEMORY_MARKER} on its own, then your rewritten memory "
    "note, folding in anything new worth remembering from the recent conversation: "
    "what the hire is working toward, what you have taught or explained, decisions "
    "made, open threads, their preferences, and what they have struggled with. Third "
    "person, factual, concise (under 200 words). Drop greetings and small talk. If "
    "nothing is new, repeat the memory unchanged. The hire never sees this part.\n"
    f"PART 3 -- the line {_ACTION_MARKER} on its own, then ONE suggested next step "
    'as JSON {"label": short button text, "question": the message to send when the '
    "hire clicks it}, or the word none when nothing fits."
)


def stream_session(
    memory: str | None,
    recent: list[Message],
    state: str,
    llm: LLMClient,
) -> Iterator[dict[str, object]]:
    """Stream the greeting as the model writes it, then yield the folded memory.

    ### Why this exists next to :func:`open_session`

    Opening the buddy took about thirty seconds, and the reason was ordering rather
    than model speed: :func:`open_session` asks for strict JSON whose **first** field
    is a memory note of up to 200 words that **the hire never sees**, with the 2-4
    sentence greeting after it. So the hire waited for roughly 260 invisible tokens
    before the first word addressed to them was even generated.

    ⚠️ **Strict JSON cannot be streamed as prose** -- the first tokens are
    ``{"memory": "`` -- so this call uses markers instead and puts the visible part
    first. Same single model call, same tokens, and the greeting starts arriving
    immediately.

    ### Degrading

    A model that ignores the format and just writes prose yields all of it as the
    greeting and **leaves the memory untouched**, which is the safe direction: a visit
    with an un-updated memory is ordinary, a memory overwritten with a greeting is not.
    An unavailable model yields the plain welcome, exactly as the non-streaming path
    does -- opening a visit must never fail the page.

    @param memory: The mentor's durable note, or None on a first visit.
    @param recent: The window since the memory was last updated.
    @param state: A snapshot of the hire's current state.
    @return: ``token`` events carrying the greeting as it arrives, then one terminal
        ``done`` carrying the whole greeting, the folded memory and any action.
    """
    prompt = [
        Message(role="system", content=_STREAM_SYSTEM),
        Message(
            role="user",
            content=(
                f"MEMORY:\n{memory or '(no memory yet -- first visit)'}\n\n"
                "RECENT conversation since the last memory update:\n"
                f"{_format_recent(recent)}\n\n"
                f"STATE (current):\n{state or '(no state available)'}\n\n"
                "Write the three parts."
            ),
        ),
    ]

    greeting = ""
    tail = ""
    pending = ""
    in_greeting = True

    try:
        for chunk in llm.stream(prompt):
            if not chunk:
                continue
            if not in_greeting:
                tail += chunk
                continue
            pending += chunk
            cut = pending.find(_MEMORY_MARKER)
            if cut != -1:
                # Trailing whitespace before the marker is formatting, not greeting.
                head = pending[:cut].rstrip()
                if head:
                    greeting += head
                    yield {"type": "token", "content": head}
                tail = pending[cut + len(_MEMORY_MARKER) :]
                pending = ""
                in_greeting = False
                continue
            # ⚠️ A suffix that could still grow into the marker is held back, never
            # emitted and never treated as the end of the greeting: "<<<MEM" is both a
            # partial marker and a legitimate five characters of prose, and only the
            # next chunk decides which. Holding back only that suffix is what keeps a
            # short greeting streaming instead of waiting for a fixed-size buffer.
            keep = _held_back_len(pending)
            if keep < len(pending):
                emit, pending = (
                    pending[: len(pending) - keep],
                    pending[len(pending) - keep :],
                )
                greeting += emit
                yield {"type": "token", "content": emit}
    except LLMUnavailableError:
        yield {
            "type": "done",
            "greeting": _FALLBACK_GREETING,
            "memory": memory or "",
            "action": None,
        }
        return

    # Whatever is still held back was never a marker after all, so it is prose.
    if in_greeting and pending:
        greeting += pending
        yield {"type": "token", "content": pending}

    folded, label, question = _split_tail(tail)
    yield {
        "type": "done",
        # ⚠️ Byte-identical to the concatenated tokens, deliberately. The client renders
        # the tokens and the caller persists this; if they differed, the message a hire
        # watched arrive would not be the one they see after a reload.
        "greeting": greeting if greeting.strip() else _FALLBACK_GREETING,
        # An empty or absent memory part means the model did not rewrite the note, so
        # the note stands. Never blanked by a malformed reply.
        "memory": folded or (memory or ""),
        "action": (
            {"label": label, "question": question} if label and question else None
        ),
    }


def _held_back_len(text: str) -> int:
    """How many trailing characters could still turn out to be the memory marker."""
    for size in range(min(len(_MEMORY_MARKER) - 1, len(text)), 0, -1):
        if _MEMORY_MARKER.startswith(text[-size:]):
            return size
    return 0


def _split_tail(tail: str) -> tuple[str, str | None, str | None]:
    """Split everything after the memory marker into the note and the action."""
    cut = tail.find(_ACTION_MARKER)
    if cut == -1:
        return tail.strip(), None, None
    memory = tail[:cut].strip()
    action = _loads_object(tail[cut + len(_ACTION_MARKER) :])
    if action is None:
        return memory, None, None
    label, question = _read_action(action)
    return memory, label, question


def _parse(raw: str, fallback_memory: str) -> BuddyOpening:
    data = _loads_object(raw)
    if data is None:
        return BuddyOpening(memory=fallback_memory, greeting=_FALLBACK_GREETING)

    memory = data.get("memory")
    greeting = data.get("greeting")
    label, question = _read_action(data.get("action"))

    return BuddyOpening(
        memory=memory
        if isinstance(memory, str) and memory.strip()
        else fallback_memory,
        greeting=(
            greeting
            if isinstance(greeting, str) and greeting.strip()
            else _FALLBACK_GREETING
        ),
        action_label=label,
        action_question=question,
    )


def _read_action(action: object) -> tuple[str | None, str | None]:
    if not isinstance(action, dict):
        return None, None
    action_dict = cast("dict[str, object]", action)
    label = action_dict.get("label")
    question = action_dict.get("question")
    has_label = isinstance(label, str) and label.strip()
    has_question = isinstance(question, str) and question.strip()
    if has_label and has_question:
        return cast("str", label), cast("str", question)
    return None, None


def _loads_object(raw: str) -> dict[str, object] | None:
    # Models sometimes wrap JSON in prose or code fences; take the outermost object.
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed: object = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else None


__all__ = ["BuddyOpening", "open_session", "stream_session"]

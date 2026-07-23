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
    "hire's current STATE (pull requests, tasks, competencies).\n"
    "Return STRICT JSON with three fields:\n"
    '1. "memory": rewrite your memory note, folding in anything new worth '
    "remembering from the recent conversation — what the hire is working toward, "
    "what you have taught or explained, decisions made, open threads, their "
    "preferences, and what they have struggled with. Third person, factual, concise "
    "(under 200 words). Drop greetings and small talk. If nothing is new, return the "
    "memory unchanged.\n"
    '2. "greeting": a short, warm, first-person opener (2-4 sentences) that greets '
    "the hire and proactively says the one thing most worth saying right now — "
    "grounded in the memory and the current state (a closed or waiting pull request, "
    "a merge to celebrate, a stall, an open thread from last time). Be specific, not "
    "generic. Never invent facts that are not in the memory or the state.\n"
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


__all__ = ["BuddyOpening", "open_session"]

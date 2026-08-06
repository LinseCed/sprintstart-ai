"""Tests for opening a buddy visit: memory fold + greeting."""

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from onboarding.buddy_open import open_session, stream_session


class _StubLLM(LLMClient):
    def __init__(self, reply: str | Exception) -> None:
        self._reply = reply
        self.last_prompt: list[Message] | None = None

    def generate(self, messages, *, temperature=None):  # type: ignore[override]
        self.last_prompt = messages
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    def chat(self, messages, tools=None):  # pragma: no cover - unused
        raise NotImplementedError

    def stream(self, messages):  # pragma: no cover - unused
        raise NotImplementedError

    def embed(self, text):  # pragma: no cover - unused
        raise NotImplementedError

    def embed_batch(self, texts):  # pragma: no cover - unused
        raise NotImplementedError

    def caption_image(self, image_bytes):  # pragma: no cover - unused
        raise NotImplementedError

    @property
    def model_name(self):  # pragma: no cover - unused
        return "stub"


def test_parses_memory_greeting_and_action() -> None:
    llm = _StubLLM(
        '{"memory": "Sam is working toward a first merge; struggled with Keycloak.",'
        ' "greeting": "Welcome back, Sam! Your PR was closed -- want a fresh task?",'
        ' "action": {"label": "Find me a task", "question": "What should I work on?"}}'
    )

    opening = open_session(memory="old note", recent=[], state="2 closed PRs", llm=llm)

    assert "first merge" in opening.memory
    assert opening.greeting.startswith("Welcome back, Sam!")
    assert opening.action_label == "Find me a task"
    assert opening.action_question == "What should I work on?"


def test_tolerates_json_wrapped_in_prose_or_fences() -> None:
    llm = _StubLLM(
        'Sure!\n```json\n{"memory": "m", "greeting": "hi there", "action": null}\n```'
    )

    opening = open_session(memory=None, recent=[], state="", llm=llm)

    assert opening.greeting == "hi there"
    assert opening.action_label is None


def test_degrades_to_prior_memory_and_plain_greeting_when_model_unavailable() -> None:
    llm = _StubLLM(LLMUnavailableError("down"))

    opening = open_session(memory="keep me", recent=[], state="", llm=llm)

    assert opening.memory == "keep me"
    assert opening.greeting  # a non-empty fallback greeting
    assert opening.action_label is None


def test_unparseable_output_keeps_prior_memory() -> None:
    llm = _StubLLM("not json at all")

    opening = open_session(memory="keep me", recent=[], state="", llm=llm)

    assert opening.memory == "keep me"
    assert opening.greeting


def test_folds_recent_conversation_into_the_prompt() -> None:
    llm = _StubLLM('{"memory": "m", "greeting": "g", "action": null}')
    recent = [
        Message(role="user", content="how do I run the tests?"),
        Message(role="assistant", content="use ./gradlew test"),
    ]

    open_session(memory=None, recent=recent, state="", llm=llm)

    assert llm.last_prompt is not None
    user_prompt = llm.last_prompt[-1]["content"]
    assert "how do I run the tests?" in user_prompt
    assert "use ./gradlew test" in user_prompt


class _StreamingStubLLM(_StubLLM):
    """Replays a scripted list of chunks, so chunk boundaries can be chosen per test."""

    def __init__(self, chunks: list[str] | Exception) -> None:
        super().__init__("")
        self._chunks = chunks

    def stream(self, messages):  # type: ignore[override]
        self.last_prompt = messages
        if isinstance(self._chunks, Exception):
            raise self._chunks
        yield from self._chunks


def _tokens(events: list[dict[str, object]]) -> str:
    return "".join(str(e["content"]) for e in events if e["type"] == "token")


def _done(events: list[dict[str, object]]) -> dict[str, object]:
    assert events[-1]["type"] == "done"
    return events[-1]


def test_streams_the_greeting_and_never_the_memory_note() -> None:
    llm = _StreamingStubLLM(
        [
            "Welcome back, Sam! ",
            "Your PR landed.",
            "\n<<<MEMORY>>>\n",
            "Sam merged their first PR.",
            "\n<<<ACTION>>>\n",
            '{"label": "What next?", "question": "What should I work on?"}',
        ]
    )

    events = list(stream_session(memory="old", recent=[], state="", llm=llm))

    # ⚠️ The whole point: the private note must never reach the hire as a token.
    assert _tokens(events) == "Welcome back, Sam! Your PR landed."
    assert "Sam merged" not in _tokens(events)
    done = _done(events)
    assert done["greeting"] == "Welcome back, Sam! Your PR landed."
    assert done["memory"] == "Sam merged their first PR."
    assert done["action"] == {
        "label": "What next?",
        "question": "What should I work on?",
    }


def test_the_greeting_starts_arriving_before_the_marker_is_reached() -> None:
    """The reason the whole change exists: first token out before generation ends."""
    llm = _StreamingStubLLM(["Hi Sam!", " Nice to see you.", "\n<<<MEMORY>>>\nnote"])

    events = list(stream_session(memory=None, recent=[], state="", llm=llm))

    assert events[0] == {"type": "token", "content": "Hi Sam!"}


def test_holds_back_only_a_possible_marker_prefix() -> None:
    """⚠️ "<<<MEM" is both a partial marker and legitimate prose.

    Only the next chunk decides which, so it must be a candidate, never an early return.
    """
    llm = _StreamingStubLLM(["Hello there<<<MEM", "ORY>>>\nthe note"])

    events = list(stream_session(memory=None, recent=[], state="", llm=llm))

    assert _tokens(events) == "Hello there"
    assert _done(events)["memory"] == "the note"


def test_a_prefix_that_turns_out_to_be_prose_is_emitted_after_all() -> None:
    llm = _StreamingStubLLM(["Careful with <<<angle", " brackets>>> in code."])

    events = list(stream_session(memory="keep me", recent=[], state="", llm=llm))

    assert _tokens(events) == "Careful with <<<angle brackets>>> in code."
    # No marker ever arrived, so there was nothing to fold: the note stands.
    assert _done(events)["memory"] == "keep me"


def test_a_marker_split_across_chunks_is_still_found() -> None:
    llm = _StreamingStubLLM(["Hi!\n<<<", "MEM", "ORY", ">>>\nthe note"])

    events = list(stream_session(memory=None, recent=[], state="", llm=llm))

    assert _tokens(events).strip() == "Hi!"
    assert _done(events)["memory"] == "the note"


def test_a_model_that_ignores_the_format_leaves_the_memory_untouched() -> None:
    """The safe direction: an un-updated note is ordinary.

    A note overwritten with a greeting is not.
    """
    llm = _StreamingStubLLM(["Just some prose with no markers at all."])

    events = list(stream_session(memory="keep me", recent=[], state="", llm=llm))

    assert _tokens(events) == "Just some prose with no markers at all."
    assert _done(events)["memory"] == "keep me"


def test_missing_action_section_is_no_action_rather_than_a_failure() -> None:
    llm = _StreamingStubLLM(["Hi!\n<<<MEMORY>>>\nthe note"])

    assert _done(list(stream_session(None, [], "", llm)))["action"] is None


def test_unparseable_action_json_is_dropped_without_losing_the_memory() -> None:
    llm = _StreamingStubLLM(["Hi!\n<<<MEMORY>>>\nthe note\n<<<ACTION>>>\nnone"])

    done = _done(list(stream_session(None, [], "", llm)))

    assert done["memory"] == "the note"
    assert done["action"] is None


def test_an_unavailable_model_still_yields_a_usable_opening() -> None:
    llm = _StreamingStubLLM(LLMUnavailableError("down"))

    done = _done(list(stream_session(memory="keep me", recent=[], state="", llm=llm)))

    assert done["greeting"]
    assert done["memory"] == "keep me"


def test_a_blank_greeting_falls_back_rather_than_showing_nothing() -> None:
    llm = _StreamingStubLLM(["\n<<<MEMORY>>>\nthe note"])

    assert _done(list(stream_session(None, [], "", llm)))["greeting"]


def test_the_streamed_prompt_carries_the_recent_window_too() -> None:
    llm = _StreamingStubLLM(["hi\n<<<MEMORY>>>\nnote"])
    recent = [Message(role="user", content="how do I run the tests?")]

    list(stream_session(memory=None, recent=recent, state="", llm=llm))

    assert llm.last_prompt is not None
    assert "how do I run the tests?" in llm.last_prompt[-1]["content"]


def test_the_done_greeting_is_exactly_what_was_streamed() -> None:
    """⚠️ The client renders the tokens; the caller persists ``done``.

    They must not differ.
    """
    llm = _StreamingStubLLM(["Hi Sam!", " Nice work.", "\n\n<<<MEMORY>>>\nnote"])

    events = list(stream_session(None, [], "", llm))

    assert _done(events)["greeting"] == _tokens(events)

"""Tests for opening a buddy visit: memory fold + greeting."""

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from onboarding.buddy_open import open_session


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

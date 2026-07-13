import json

from insights.redaction import redact_pii
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient


class _NameRedactingLLM(StubLLMClient):
    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        payload = json.loads(messages[-1]["content"])  # type: ignore[index]
        redacted = [t.replace("John Doe", "[NAME]") for t in payload["texts"]]
        return json.dumps({"texts": redacted})


class _FailingLLM(StubLLMClient):
    def generate(self, messages: list[dict[str, object]]) -> str:
        raise LLMUnavailableError("local LLM unavailable")


class _GarbageLLM(StubLLMClient):
    def generate(self, messages: list[dict[str, object]]) -> str:
        return "not json"


class _WrongLengthLLM(StubLLMClient):
    def generate(self, messages: list[dict[str, object]]) -> str:
        return json.dumps({"texts": ["only one"]})


class _EchoLLM(StubLLMClient):
    """Returns exactly what it was sent, so regex-stage output is observable."""

    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        payload = json.loads(messages[-1]["content"])  # type: ignore[index]
        return json.dumps({"texts": payload["texts"]})


def test_redact_pii_empty_input_returns_empty() -> None:
    assert redact_pii([], StubLLMClient()) == []


def test_redact_pii_redacts_email_and_phone_via_regex() -> None:
    result = redact_pii(
        ["Contact me at jane@example.com or +1 555-123-4567"],
        _EchoLLM(),
    )

    assert result == ["Contact me at [EMAIL] or [PHONE]"]


def test_redact_pii_applies_llm_name_redaction() -> None:
    result = redact_pii(["Ask John Doe for VPN access"], _NameRedactingLLM())

    assert result == ["Ask [NAME] for VPN access"]


def test_redact_pii_falls_back_to_regex_only_when_llm_unavailable() -> None:
    result = redact_pii(
        ["Contact me at jane@example.com"],
        _FailingLLM(),
    )

    assert result == ["Contact me at [EMAIL]"]


def test_redact_pii_falls_back_to_regex_only_on_unparsable_output() -> None:
    result = redact_pii(["Ask John Doe for help"], _GarbageLLM())

    assert result == ["Ask John Doe for help"]


def test_redact_pii_falls_back_to_regex_only_on_length_mismatch() -> None:
    result = redact_pii(["first question", "second question"], _WrongLengthLLM())

    assert result == ["first question", "second question"]

from collections.abc import Iterator

from llm.base import Message


class StubLLMClient:
    def __init__(
        self,
        generate_response: str = "stub answer",
        embedding: list[float] | None = None,
        caption: str = "stub caption",
    ) -> None:
        self.generate_response = generate_response
        self.embedding = embedding or [0.0] * 768
        self.caption = caption

    def generate(self, messages: list[Message]) -> str:
        return self.generate_response

    def stream(self, messages: list[Message]) -> Iterator[str]:
        yield self.generate_response

    def embed(self, text: str) -> list[float]:
        return self.embedding

    def caption_image(self, image_bytes: bytes) -> str:
        return self.caption


class ScriptedLLMClient:
    def __init__(
        self,
        responses: list[str],
        *,
        answer: str = "final answer",
        embedding: list[float] | None = None,
    ) -> None:
        self._responses = list(responses)
        self.answer = answer
        self.embedding = embedding or [0.0] * 768
        self.generate_calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> str:
        self.generate_calls.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return "READY"

    def stream(self, messages: list[Message]) -> Iterator[str]:
        yield self.answer

    def embed(self, text: str) -> list[float]:
        return self.embedding

    def caption_image(self, image_bytes: bytes) -> str:
        return "stub caption"

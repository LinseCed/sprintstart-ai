from collections.abc import Iterator, Mapping, Sequence

from llm.base import ChatResult, Message, ToolCall, ToolSpec

Turn = Sequence[tuple[str, Mapping[str, object]]]


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

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult:
        return ChatResult(text=self.generate_response, tool_calls=[])

    def generate(self, messages: list[Message]) -> str:
        return self.generate_response

    def stream(self, messages: list[Message]) -> Iterator[str]:
        yield self.generate_response

    def embed(self, text: str) -> list[float]:
        return self.embedding

    def caption_image(self, image_bytes: bytes) -> str:
        return self.caption


class ScriptedLLMClient:
    """
    Drives the tool-calling loop deterministically.

    Each `chat` call pops the next scripted turn and returns those tool calls. Once
    the script is exhausted it returns no tool calls (the agent stops gathering).
    `stream` always yields the fixed answer.
    """

    def __init__(
        self,
        turns: Sequence[Turn],
        *,
        answer: str = "final answer",
        embedding: list[float] | None = None,
    ) -> None:
        self._turns: list[Turn] = list(turns)
        self.answer = answer
        self.embedding = embedding or [0.0] * 768
        self.chat_calls: list[list[Message]] = []
        self.stream_calls: list[list[Message]] = []

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult:
        self.chat_calls.append(messages)
        turn: Turn = self._turns.pop(0) if self._turns else []
        calls = [
            ToolCall(id=f"call_{i}", name=name, arguments=dict(args))
            for i, (name, args) in enumerate(turn)
        ]
        return ChatResult(text="" if calls else self.answer, tool_calls=calls)

    def generate(self, messages: list[Message]) -> str:
        return self.answer

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self.stream_calls.append(messages)
        yield self.answer

    def embed(self, text: str) -> list[float]:
        return self.embedding

    def caption_image(self, image_bytes: bytes) -> str:
        return "stub caption"

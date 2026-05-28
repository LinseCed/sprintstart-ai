from collections.abc import Iterator
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


class LLMClient(Protocol):
    def generate(self, messages: list[Message]) -> str: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...
    def embed(self, text: str) -> list[float]: ...

from collections.abc import Iterator, Sequence
from typing import Protocol

import ollama

from llm.base import Message
from llm.errors import LLMUnavailableError


class OllamaBackend(Protocol):
    def chat(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
    ) -> ollama.ChatResponse: ...

    def chat_stream(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
    ) -> Iterator[ollama.ChatResponse]: ...

    def embeddings(
        self,
        model: str = "",
        prompt: str = "",
    ) -> ollama.EmbeddingsResponse: ...

    def chat_with_images(
        self,
        model: str = "",
        prompt: str = "",
        images: list[bytes] | None = None,
    ) -> ollama.ChatResponse: ...


class _OllamaAdapter:
    """Wraps ollama.Client to satisfy the OllamaBackend protocol."""

    def __init__(self, host: str | None) -> None:
        self._client = ollama.Client(host=host)

    def chat(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
    ) -> ollama.ChatResponse:
        # ollama.Client.chat uses @overload and pyright cannot resolve the member type
        return self._client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=model, messages=list(messages or [])
        )

    def chat_stream(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
    ) -> Iterator[ollama.ChatResponse]:
        # stream=True makes ollama return Iterator[ChatResponse] at runtime, but the
        # @overload stubs are too coarse for pyright to narrow the return type
        return self._client.chat(  # type: ignore[return-value]
            model=model, messages=list(messages or []), stream=True
        )

    def embeddings(
        self,
        model: str = "",
        prompt: str = "",
    ) -> ollama.EmbeddingsResponse:
        return self._client.embeddings(model=model, prompt=prompt)

    def chat_with_images(
        self,
        model: str = "",
        prompt: str = "",
        images: list[bytes] | None = None,
    ) -> ollama.ChatResponse:
        return self._client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=model,
            messages=[{"role": "user", "content": prompt, "images": images or []}],
        )


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        vision_model: str | None = None,
        client: OllamaBackend | None = None,
    ) -> None:
        self._host = host
        self._model = model
        self._embed_model = embed_model
        self._vision_model = vision_model
        self._client: OllamaBackend = (
            client if client is not None else _OllamaAdapter(host=host)
        )

    def generate(self, messages: list[Message]) -> str:
        if self._model is None:
            raise ValueError("No model specified")
        try:
            response = self._client.chat(model=self._model, messages=messages)
            return response.message.content or ""
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

    def stream(self, messages: list[Message]) -> Iterator[str]:
        if self._model is None:
            raise ValueError("No model specified")
        try:
            for chunk in self._client.chat_stream(model=self._model, messages=messages):
                yield chunk.message.content or ""
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

    def embed(self, text: str) -> list[float]:
        if self._embed_model is None:
            raise ValueError("No embed model specified")
        try:
            response = self._client.embeddings(model=self._embed_model, prompt=text)
            return list(response.embedding or [])
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

    def caption_image(self, image_bytes: bytes) -> str:
        if self._vision_model is None:
            raise LLMUnavailableError(self._host)
        try:
            response = self._client.chat_with_images(
                model=self._vision_model,
                prompt="Describe this image concisely.",
                images=[image_bytes],
            )
            return response.message.content or ""
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

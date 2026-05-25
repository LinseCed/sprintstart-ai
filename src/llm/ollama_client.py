from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import ollama

from llm.errors import LLMUnavailableError


class OllamaBackend(Protocol):
    def chat(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, str]] | None = None,
    ) -> ollama.ChatResponse: ...

    def embeddings(
        self,
        model: str = "",
        prompt: str = "",
    ) -> ollama.EmbeddingsResponse: ...


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        client: OllamaBackend | None = None,
    ) -> None:
        self._host = host
        self._model = model
        self._embed_model = embed_model
        self._client: OllamaBackend = (
            client
            if client is not None
            else cast(OllamaBackend, ollama.Client(host=host))
        )

    def generate(self, prompt: str) -> str:
        if self._model is None:
            raise ValueError("No model specified")

        messages = [{"role": "user", "content": prompt}]
        try:
            response = self._client.chat(model=self._model, messages=messages)
            return response.message.content or ""
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

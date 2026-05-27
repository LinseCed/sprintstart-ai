import os
from collections.abc import Iterator, Mapping, Sequence

import httpx
import ollama
import pytest

from llm.base import Message
from llm.errors import LLMUnavailableError
from llm.ollama_client import OllamaBackend, OllamaClient

_LOCAL_HOST = os.environ.get("OLLAMA_HOST")
_TEST_MODEL = os.environ.get("OLLAMA_MODEL", "test-model")
_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "test-embed-model")
_OLLAMA_BASE = _LOCAL_HOST or "http://localhost:11434"


def _ollama_is_reachable() -> bool:
    try:
        httpx.get(_OLLAMA_BASE, timeout=2)
        return True
    except httpx.ConnectError:
        return False


ollama_required = pytest.mark.skipif(
    not _ollama_is_reachable(), reason="Ollama not reachable - skipping"
)


class _FakeOllamaClient:
    def __init__(
        self,
        chat_content: str = "",
        stream_tokens: list[str] | None = None,
        embed_vector: list[float] | None = None,
        chat_error: Exception | None = None,
        embed_error: Exception | None = None,
    ) -> None:
        self._chat_content = chat_content
        self._stream_tokens = stream_tokens or [chat_content]
        self._embed_vector = embed_vector or []
        self._chat_error = chat_error
        self._embed_error = embed_error
        self.last_chat_model: str | None = None
        self.last_chat_messages: list[Mapping[str, str]] | None = None
        self.last_embed_model: str | None = None
        self.last_embed_prompt: str | None = None

    def chat(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, str]] | None = None,
    ) -> ollama.ChatResponse:
        self.last_chat_model = model
        self.last_chat_messages = list(messages) if messages is not None else None
        if self._chat_error is not None:
            raise self._chat_error
        return ollama.ChatResponse(
            message=ollama.Message(role="assistant", content=self._chat_content)
        )

    def chat_stream(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, str]] | None = None,
    ) -> Iterator[ollama.ChatResponse]:
        self.last_chat_model = model
        self.last_chat_messages = list(messages) if messages is not None else None
        if self._chat_error is not None:
            raise self._chat_error
        for token in self._stream_tokens:
            yield ollama.ChatResponse(
                message=ollama.Message(role="assistant", content=token)
            )

    def embeddings(
        self,
        model: str = "",
        prompt: str = "",
    ) -> ollama.EmbeddingsResponse:
        self.last_embed_model = model
        self.last_embed_prompt = prompt
        if self._embed_error is not None:
            raise self._embed_error
        return ollama.EmbeddingsResponse(embedding=self._embed_vector)


def _make_client(
    host: str | None = _LOCAL_HOST,
    model: str | None = _TEST_MODEL,
    embed_model: str | None = _EMBED_MODEL,
    inner_client: OllamaBackend | None = None,
) -> OllamaClient:
    return OllamaClient(
        host=host, model=model, embed_model=embed_model, client=inner_client
    )


class TestGenerateHappyPath:
    def test_returns_assistant_content(self) -> None:
        fake = _FakeOllamaClient(chat_content="Hello there!")
        client = _make_client(inner_client=fake)
        messages = [Message(role="user", content="Say hello")]

        result = client.generate(messages)

        assert result == "Hello there!"
        assert fake.last_chat_model == _TEST_MODEL
        assert fake.last_chat_messages == [{"role": "user", "content": "Say hello"}]


@ollama_required
class TestGenerateIntegration:
    def test_returns_a_string(self) -> None:
        client = _make_client()
        result = client.generate([Message(role="user", content="Reply with one word: hello")])
        assert isinstance(result, str)
        assert len(result) > 0


class TestStreamHappyPath:
    def test_yields_tokens(self) -> None:
        fake = _FakeOllamaClient(stream_tokens=["Hello", " there", "!"])
        client = _make_client(inner_client=fake)
        messages = [Message(role="user", content="Say hello")]

        result = list(client.stream(messages))

        assert result == ["Hello", " there", "!"]
        assert fake.last_chat_model == _TEST_MODEL

    def test_yields_single_token(self) -> None:
        fake = _FakeOllamaClient(chat_content="Hi!")
        client = _make_client(inner_client=fake)

        result = list(client.stream([Message(role="user", content="Hi")]))

        assert result == ["Hi!"]


@ollama_required
class TestStreamIntegration:
    def test_yields_strings(self) -> None:
        client = _make_client()
        tokens = list(client.stream([Message(role="user", content="Reply with one word: hello")]))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)


class TestEmbedHappyPath:
    def test_returns_vector(self) -> None:
        fake_vector = [0.1, 0.2, 0.3]
        fake = _FakeOllamaClient(embed_vector=fake_vector)
        client = _make_client(inner_client=fake)

        result = client.embed("Say hello")

        assert result == fake_vector
        assert fake.last_embed_model == _EMBED_MODEL
        assert fake.last_embed_prompt == "Say hello"


@ollama_required
class TestEmbedIntegration:
    def test_returns_a_float_vector(self) -> None:
        client = _make_client()
        result = client.embed("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(v, float) for v in result)


class TestLLMUnavailableError:
    def test_generate_raises_on_connection_error(self) -> None:
        fake = _FakeOllamaClient(chat_error=ConnectionError("refused"))
        client = _make_client(host="http://localhost:1", inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.generate([Message(role="user", content="hello")])

    def test_stream_raises_on_connection_error(self) -> None:
        fake = _FakeOllamaClient(chat_error=ConnectionError("refused"))
        client = _make_client(host="http://localhost:1", inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            list(client.stream([Message(role="user", content="hello")]))

    def test_embed_raises_on_connection_error(self) -> None:
        fake = _FakeOllamaClient(embed_error=ConnectionError("refused"))
        client = _make_client(host="http://localhost:1", inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.embed("hello")

    def test_generate_raises_on_ollama_response_error(self) -> None:
        fake = _FakeOllamaClient(chat_error=ollama.ResponseError("model not found"))
        client = _make_client(inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.generate([Message(role="user", content="hello")])

    def test_embed_raises_on_ollama_response_error(self) -> None:
        fake = _FakeOllamaClient(embed_error=ollama.ResponseError("model not found"))
        client = _make_client(inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.embed("hello")

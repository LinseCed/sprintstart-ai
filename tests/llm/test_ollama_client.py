import os

import httpx
import ollama
import pytest
from unittest.mock import patch, MagicMock

from llm.ollama_client import OllamaClient
from llm.errors import LLMUnavailableError

backend = os.environ.get("LLM_BACKEND")

match backend:
    case "ollama":
        _LOCAL_HOST = os.environ.get("OLLAMA_HOST")
        _TEST_MODEL = os.environ.get("OLLAMA_MODEL")
        _EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL")
    case _:
        raise ValueError(f"Unknown LLM_BACKEND {backend!r}")

def _ollama_is_reachable() -> bool:
    try:
        httpx.get("http://localhost:11434", timeout=2)
        return True
    except httpx.ConnectError:
        return False


ollama_required = pytest.mark.skipif(
    not _ollama_is_reachable(), reason="Ollama not reachable - skipping"
)


def _make_client(
    host: str = _LOCAL_HOST, model: str = _TEST_MODEL, embed_model: str = _EMBED_MODEL
) -> OllamaClient:
    return OllamaClient(host=host, model=model, embed_model=embed_model)


class TestGenerateHappyPath:
    def test_returns_assistant_content(self) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.message.content = "Hello there!"
        with patch.object(
            client._client, "chat", return_value=mock_response
        ) as mock_chat:
            result = client.generate("Say hello")
            assert result == "Hello there!"
            mock_chat.assert_called_once_with(
                model=_TEST_MODEL,
                messages=[{"role": "user", "content": "Say hello"}],
            )

@ollama_required
class TestGenerateIntegration:
    def test_returns_a_string(self) -> None:
        client = _make_client()
        result = client.generate("Reply with one word: hello")
        assert isinstance(result, str)
        assert len(result) > 0

class TestEmbedHappyPath:
    def test_returns_vector(self) -> None:
        client = _make_client()
        fake_vector = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.embedding = fake_vector
        with patch.object(client._client, "embeddings", return_value=mock_response) as mock_chat:
            result = client.embed("Say hello")
            assert result == fake_vector
            mock_chat.assert_called_once_with(model=_EMBED_MODEL, prompt="Say hello")


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
        client = _make_client(host="http://localhost:1")
        with patch.object(
            client._client,
            "chat",
            side_effect=ConnectionError("refused"),
        ):
            with pytest.raises(LLMUnavailableError):
                client.generate("hello")

    def test_embed_raises_on_connection_error(self) -> None:
        client = _make_client(host="http://localhost:1")
        with patch.object(
            client._client,
            "embeddings",
            side_effect=ConnectionError("refused"),
        ):
            with pytest.raises(LLMUnavailableError):
                client.embed("hello")

    def test_generate_raises_on_ollama_response_error(self) -> None:
        client = _make_client()
        with patch.object(
            client._client,
            "chat",
            side_effect=ollama.ResponseError("model not found"),
        ):
            with pytest.raises(LLMUnavailableError):
                client.generate("hello")

    def test_embed_raises_on_ollama_response_error(self) -> None:
        client = _make_client()
        with patch.object(
            client._client,
            "embeddings",
            side_effect=ollama.ResponseError("model not found"),
        ):
            with pytest.raises(LLMUnavailableError):
                client.embed("hello")
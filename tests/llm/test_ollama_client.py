import os

import httpx
import pytest
from unittest.mock import patch, MagicMock

from src.llm.ollama_client import OllamaClient

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
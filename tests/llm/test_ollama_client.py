import httpx
import pytest
from unittest.mock import patch, MagicMock

from src.llm.ollama_client import OllamaClient

_LOCAL_HOST = "http://localhost:11434"
_TEST_MODEL = "llama3.2:1b"
_EMBED_MODEL = "nomic-embed-text"


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

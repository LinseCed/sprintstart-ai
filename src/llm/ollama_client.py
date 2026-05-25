import ollama

from llm.errors import LLMUnavailableError


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self._host = host
        self._model = model
        self._embed_model = embed_model
        self._client = ollama.Client(host=self._host)

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
            return response.embedding or []
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

import os
from functools import lru_cache

from llm.base import LLMClient
from llm.ollama_client import OllamaClient
from store.base import VectorStore
from store.chroma_store import ChromaVectorStore


@lru_cache
def get_llm() -> LLMClient:
    backend = os.getenv("LLM_BACKEND", "ollama")

    if backend == "ollama":
        return OllamaClient(
            host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        )

    raise ValueError(f"Unknown LLM backend: {backend!r}")


@lru_cache
def get_store() -> VectorStore:
    return ChromaVectorStore(path=os.getenv("CHROMA_PATH"))

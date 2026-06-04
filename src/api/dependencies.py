import logging
import os
from functools import lru_cache

from llm.base import LLMClient
from llm.ollama_client import OllamaClient
from store.base import VectorStore
from store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> LLMClient:
    backend = os.getenv("LLM_BACKEND")

    if backend == "ollama":
        return OllamaClient(
            host=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_MODEL"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL"),
            vision_model=os.getenv("OLLAMA_VISION_MODEL"),
        )

    raise ValueError(f"Unknown LLM backend: {backend!r}")


@lru_cache
def get_store() -> VectorStore:
    path = os.getenv("CHROMA_PATH")
    if path is None:
        logger.warning(
            "CHROMA_PATH is not set — using ephemeral in-memory store, "
            "data will not persist"
        )
    return ChromaVectorStore(path=path)

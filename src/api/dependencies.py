import logging
import os
from functools import lru_cache

from fastapi import Depends

from agents.orchestrator import ChatOrchestrator
from llm.base import LLMClient
from llm.ollama_client import OllamaClient
from store.base import VectorStore
from store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> LLMClient:
    backend = (os.getenv("LLM_BACKEND") or os.getenv("LLM_PROVIDER") or "local").lower()

    if backend in {"ollama", "local"}:
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


def get_orchestrator(
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> ChatOrchestrator:
    return ChatOrchestrator(llm, store)

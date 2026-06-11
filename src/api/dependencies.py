import logging
import os
from functools import lru_cache

from fastapi import Depends

from agents.orchestrator import ChatOrchestrator
from llm.base import LLMClient
from llm.ollama_client import OllamaClient
from llm.openai_client import OpenAIClient
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

    if backend in {"openai", "openai-compatible"}:
        return OpenAIClient(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY") or "unused",
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            vision_model=os.getenv("OPENAI_VISION_MODEL"),
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

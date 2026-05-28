import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _llm_is_reachable() -> bool:
    try:
        httpx.get(_OLLAMA_BASE, timeout=2)
        return True
    except httpx.HTTPError:
        return False


llm_required = pytest.mark.skipif(
    not _llm_is_reachable(), reason="LLM backend not reachable - skipping"
)


@pytest.fixture(autouse=True)
def clear_dependency_caches() -> None:
    from api.dependencies import get_llm, get_store

    get_llm.cache_clear()
    get_store.cache_clear()

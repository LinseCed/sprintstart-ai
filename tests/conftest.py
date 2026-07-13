import json
import os
from pathlib import Path
from typing import Any

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

vision_required = pytest.mark.skipif(
    not _llm_is_reachable() or os.environ.get("OLLAMA_VISION_MODEL") is None,
    reason="Vision model not configured or Ollama not reachable - skipping",
)


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
    ]


@pytest.fixture(autouse=True)
def clear_dependency_caches() -> None:
    from api.dependencies import (
        get_ingestion_metadata_store,
        get_llm,
        get_source_state_store,
        get_store,
    )

    get_llm.cache_clear()
    get_store.cache_clear()
    get_ingestion_metadata_store.cache_clear()
    get_source_state_store.cache_clear()

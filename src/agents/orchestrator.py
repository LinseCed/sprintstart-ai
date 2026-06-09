from typing import Iterator

from api.schemas import HistoryEntry
from llm.base import LLMClient
from store.base import VectorStore


class ChatOrchestrator:
    def __init__(self, client: LLMClient, store: VectorStore):
        self.client = client
        self.store = store

    def stream(self, query: str, history: list[HistoryEntry]) -> Iterator[str]:
        yield "not implemented"

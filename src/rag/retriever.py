from llm.base import LLMClient
from rag.types import RetrievalFilters, ScoredChunk
from store.base import VectorStore


def retrieve(
    question: str,
    llm: LLMClient,
    store: VectorStore,
    top_k: int,
    min_score: float,
    filters: RetrievalFilters | None = None,
) -> list[ScoredChunk]:
    embedding = llm.embed(question)

    return store.query(
        embedding=embedding,
        top_k=top_k,
        min_score=min_score,
        filters=filters,
    )

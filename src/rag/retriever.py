from llm.base import LLMClient
from rag.types import ScoredChunk
from store.base import VectorStore


def retrieve(
    question: str,
    llm: LLMClient,
    store: VectorStore,
    top_k: int,
    min_score: float,
) -> list[ScoredChunk]:
    embedding = llm.embed(question)

    return store.query(
        embedding=embedding,
        top_k=top_k,
        min_score=min_score,
    )

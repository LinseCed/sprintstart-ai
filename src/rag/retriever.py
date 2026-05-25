from src.llm.base import LLMClient
from src.rag.types import Chunk
from src.store.base import VectorStore


def retrieve(
    question: str,
    llm: LLMClient,
    store: VectorStore,
    top_k: int,
    min_score: float,
) -> list[Chunk]:
    embedding = llm.embed(question)

    return store.query(
        embedding=embedding,
        top_k=top_k,
        min_score=min_score,
    )
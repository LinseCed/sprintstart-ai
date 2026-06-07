from llm.base import LLMClient
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import Chunk
from store.base import VectorStore

_BM25_CACHE = BM25IndexCache()


def retrieve(
    question: str,
    llm: LLMClient,
    store: VectorStore,
    top_k: int,
    min_score: float,
) -> list[Chunk]:
    return hybrid_retrieve(
        question=question,
        llm=llm,
        store=store,
        top_k=top_k,
        min_score=min_score,
        bm25_cache=_BM25_CACHE,
    )
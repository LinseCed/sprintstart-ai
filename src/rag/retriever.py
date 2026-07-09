from llm.base import LLMClient
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

# Process-wide BM25 index, shared across every caller (chat, agent tools, and
# onboarding) so the corpus is only tokenized once instead of once per caller.
_BM25_CACHE = BM25IndexCache()


def get_bm25_cache() -> BM25IndexCache:
    return _BM25_CACHE


def retrieve(
    question: str,
    llm: LLMClient,
    store: VectorStore,
    top_k: int,
    min_score: float,
) -> list[ScoredChunk]:
    return hybrid_retrieve(
        question=question,
        llm=llm,
        store=store,
        top_k=top_k,
        min_score=min_score,
        bm25_cache=_BM25_CACHE,
    )

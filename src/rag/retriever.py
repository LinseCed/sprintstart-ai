from llm.base import LLMClient
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.source_filter import SourceExclusions
from rag.types import ScoredChunk
from store.base import VectorStore

_BM25_CACHE = BM25IndexCache()


def retrieve(
    question: str,
    llm: LLMClient,
    store: VectorStore,
    top_k: int,
    min_score: float,
    exclusions: SourceExclusions = SourceExclusions(),
) -> list[ScoredChunk]:
    return hybrid_retrieve(
        question=question,
        llm=llm,
        store=store,
        top_k=top_k,
        min_score=min_score,
        bm25_cache=_BM25_CACHE,
        exclusions=exclusions,
    )

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from ingestion.source_role import SourceRole
from llm.base import LLMClient
from rag.types import Chunk, ScoredChunk
from store.base import VectorStore

if TYPE_CHECKING:
    from collections.abc import Sequence

RRF_K = 60
RRF_MIN_RATIO = 0.15
SEMANTIC_ONLY_CHUNK_LIMIT = 10_000

# When a role filter is active we over-fetch from each retriever so that, after
# excluded-role chunks are dropped, enough candidates remain to fill ``top_k``.
_ROLE_FILTER_OVERFETCH = 5


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())


def to_scored_chunk(chunk: Chunk | ScoredChunk, score: float) -> ScoredChunk:
    return ScoredChunk(
        id=chunk.id,
        artifact_id=chunk.artifact_id,
        filename=chunk.filename,
        position=chunk.position,
        kind=chunk.kind,
        text=chunk.text,
        score=score,
        source_role=chunk.source_role,
        source_url=chunk.source_url,
        artifact_type=chunk.artifact_type,
        language=chunk.language,
        start_line=chunk.start_line,
        start_page=chunk.start_page,
    )


def _drop_excluded_roles(
    chunks: list[ScoredChunk], exclude_roles: frozenset[SourceRole]
) -> list[ScoredChunk]:
    if not exclude_roles:
        return chunks
    return [c for c in chunks if c.source_role not in exclude_roles]


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    k: int = RRF_K,
) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, ScoredChunk] = {}

    for ranked_chunks in ranked_lists:
        for rank, chunk in enumerate(ranked_chunks, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks_by_id[chunk.id] = chunk

    fused = [
        to_scored_chunk(chunks_by_id[chunk_id], score)
        for chunk_id, score in scores.items()
    ]

    fused.sort(key=lambda chunk: chunk.score, reverse=True)
    return fused


def filter_by_rrf_ratio(
    chunks: list[ScoredChunk],
    min_ratio: float = RRF_MIN_RATIO,
) -> list[ScoredChunk]:
    if not chunks:
        return []

    top_score = chunks[0].score
    threshold = top_score * min_ratio

    return [chunk for chunk in chunks if chunk.score >= threshold]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.index: Any | None = (
            BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        )

    def query(self, question: str, top_k: int) -> list[ScoredChunk]:
        if not self.chunks or self.index is None:
            return []

        tokenized_question = tokenize(question)
        raw_scores = self.index.get_scores(tokenized_question)
        scores = cast("Sequence[float]", raw_scores)

        ranked = sorted(
            zip(self.chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            to_scored_chunk(chunk, float(score))
            for chunk, score in ranked[:top_k]
            if score > 0
        ]


class BM25IndexCache:
    def __init__(self) -> None:
        self._index: BM25Index | None = None
        self._chunk_count: int | None = None
        self._lock = threading.Lock()

    def get(self, store: VectorStore) -> BM25Index:
        with self._lock:
            current_count = store.count()

            if self._index is None or self._chunk_count != current_count:
                self._index = BM25Index(store.all_chunks())
                self._chunk_count = current_count

            return self._index


def hybrid_retrieve(
    question: str,
    llm: LLMClient,
    store: VectorStore,
    top_k: int,
    min_score: float,
    bm25_cache: BM25IndexCache,
    exclude_roles: frozenset[SourceRole] = frozenset(),
) -> list[ScoredChunk]:
    """Hybrid (semantic + BM25) retrieval with reciprocal-rank fusion.

    ``exclude_roles`` drops chunks of the given :data:`SourceRole`\\ s (e.g.
    ``"test"``) from the candidates *before* fusion. Because each retriever
    returns its own top-k, we over-fetch when a filter is active so excluded
    chunks don't starve the result. Legacy chunks without a role default to
    ``primary`` and are therefore never excluded.
    """
    chunk_count = store.count()
    # Over-fetch so role filtering still leaves enough candidates for top_k.
    fetch_k = top_k * _ROLE_FILTER_OVERFETCH if exclude_roles else top_k

    if chunk_count > SEMANTIC_ONLY_CHUNK_LIMIT:
        embedding = llm.embed(question)
        results = store.query(
            embedding=embedding,
            top_k=fetch_k,
            min_score=min_score,
        )
        return _drop_excluded_roles(results, exclude_roles)[:top_k]

    bm25_index = bm25_cache.get(store)

    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(
            lambda: store.query(
                embedding=llm.embed(question),
                top_k=fetch_k,
                min_score=min_score,
            )
        )

        bm25_future = executor.submit(
            lambda: bm25_index.query(
                question=question,
                top_k=fetch_k,
            )
        )

        semantic_results = semantic_future.result()
        bm25_results = bm25_future.result()

    semantic_results = _drop_excluded_roles(semantic_results, exclude_roles)
    bm25_results = _drop_excluded_roles(bm25_results, exclude_roles)

    if not semantic_results:
        return []

    fused = reciprocal_rank_fusion(
        ranked_lists=[semantic_results, bm25_results],
        k=RRF_K,
    )

    filtered = filter_by_rrf_ratio(
        chunks=fused,
        min_ratio=RRF_MIN_RATIO,
    )

    return filtered[:top_k]

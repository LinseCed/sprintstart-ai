from src.rag.hybrid import (
    BM25IndexCache,
    hybrid_retrieve,
    reciprocal_rank_fusion,
)
from src.rag.types import Chunk, ScoredChunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    source_role: str = "primary",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        artifact_id="artifact-1",
        filename="doc.md",
        text=text,
        embedding=embedding,
        source_role=source_role,  # type: ignore[arg-type]
    )


def make_scored_chunk(chunk_id: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        id=chunk_id,
        artifact_id="artifact-1",
        filename="doc.md",
        text=text,
        score=0.0,
    )


def test_rrf_merge_with_known_rankings() -> None:
    chunk_a = make_scored_chunk("a", "alpha")
    chunk_b = make_scored_chunk("b", "beta")
    chunk_c = make_scored_chunk("c", "gamma")

    semantic_results = [chunk_a, chunk_b]
    bm25_results = [chunk_b, chunk_c]

    result = reciprocal_rank_fusion(
        ranked_lists=[semantic_results, bm25_results],
        k=60,
    )

    assert result[0].id == "b"
    assert {chunk.id for chunk in result} == {"a", "b", "c"}


def test_bm25_cache_invalidates_when_chunk_count_changes() -> None:
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk(
                chunk_id="chunk-1",
                text="first chunk",
                embedding=[1.0, 0.0],
            )
        ]
    )

    first_index = cache.get(store)

    store.add(
        [
            make_chunk(
                chunk_id="chunk-2",
                text="second chunk",
                embedding=[0.0, 1.0],
            )
        ]
    )

    second_index = cache.get(store)

    assert first_index is not second_index


def test_large_corpus_uses_semantic_only_fallback() -> None:
    class LargeCorpusStore(StubVectorStore):
        def count(self) -> int:
            return 10_001

    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = LargeCorpusStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk(
                chunk_id="semantic-match",
                text="semantic match",
                embedding=[1.0, 0.0],
            ),
            make_chunk(
                chunk_id="keyword-only",
                text="exact SPECIAL_KEYWORD",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    result = hybrid_retrieve(
        question="SPECIAL_KEYWORD",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.8,
        bm25_cache=cache,
    )

    assert len(result) == 1
    assert result[0].id == "semantic-match"


def test_exclude_roles_drops_test_chunks() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk("primary", "onboarding setup guide", [1.0, 0.0]),
            make_chunk("test", "onboarding setup guide", [1.0, 0.0], "test"),
        ]
    )

    result = hybrid_retrieve(
        question="onboarding setup",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclude_roles=frozenset({"test"}),
    )

    ids = {chunk.id for chunk in result}
    assert ids == {"primary"}


def test_exclude_roles_keeps_legacy_unmarked_chunks() -> None:
    """Chunks without a role default to 'primary' and survive a test filter."""
    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = StubVectorStore()
    cache = BM25IndexCache()

    # Built without source_role → defaults to "primary".
    store.add([make_chunk("legacy", "onboarding setup guide", [1.0, 0.0])])

    result = hybrid_retrieve(
        question="onboarding setup",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclude_roles=frozenset({"test"}),
    )

    assert {chunk.id for chunk in result} == {"legacy"}


def test_exclude_roles_in_semantic_only_fallback() -> None:
    class LargeCorpusStore(StubVectorStore):
        def count(self) -> int:
            return 10_001

    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = LargeCorpusStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk("primary", "match", [1.0, 0.0]),
            make_chunk("test", "match", [1.0, 0.0], "test"),
        ]
    )

    result = hybrid_retrieve(
        question="match",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclude_roles=frozenset({"test"}),
    )

    assert {chunk.id for chunk in result} == {"primary"}

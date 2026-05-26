from rag.retriever import retrieve
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def test_retrieve_returns_chunks_above_min_score() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])

    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                heading_path="Intro",
                text="Relevant text",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-1",
                filename="doc.md",
                heading_path="Other",
                text="Irrelevant text",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    result = retrieve(
        question="What is this about?",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.8,
    )

    assert len(result) == 1
    assert result[0].id == "chunk-1"


def test_retrieve_returns_empty_list_when_no_chunk_passes_threshold() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])

    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                heading_path="Intro",
                text="Not similar",
                embedding=[0.0, 1.0],
            )
        ]
    )

    result = retrieve(
        question="What is this about?",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.8,
    )

    assert result == []

import chromadb

from src.rag.types import Chunk
from src.store.chroma_store import ChromaVectorStore


def test_chroma_query_returns_chunks_above_min_score() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_query",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                heading_path="Intro",
                position=1,
                kind="text",
                text="Relevant text",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-1",
                filename="doc.md",
                heading_path="Other",
                position=2,
                kind="text",
                text="Irrelevant text",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.8,
    )

    assert len(result) == 1
    assert result[0].id == "chunk-1"
    assert result[0].score is not None
    assert result[0].score >= 0.8


def test_chroma_query_returns_empty_list_when_threshold_too_high() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_empty",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Some text",
                embedding=[0.0, 1.0],
            )
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.8,
    )

    assert result == []


def test_chroma_delete_removes_only_matching_artifact() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_delete",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="a.md",
                text="Text A",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-2",
                filename="b.md",
                text="Text B",
                embedding=[1.0, 0.0],
            ),
        ]
    )

    store.delete("artifact-1")

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.1,
    )

    assert len(result) == 1
    assert result[0].id == "chunk-2"
    assert result[0].artifact_id == "artifact-2"

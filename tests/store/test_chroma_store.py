from pathlib import Path

import chromadb

from rag.types import Chunk
from store.chroma_store import ChromaVectorStore


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
                position=1,
                kind="text",
                text="Relevant text",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-1",
                filename="doc.md",
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


def test_chroma_query_returns_empty_list_when_collection_is_empty() -> None:
    store = ChromaVectorStore(collection_name="test_empty_collection")

    result = store.query(embedding=[1.0, 0.0], top_k=5, min_score=0.0)

    assert result == []


def test_chroma_add_empty_list_is_noop() -> None:
    store = ChromaVectorStore(collection_name="test_add_empty")

    store.add([])

    result = store.query(embedding=[1.0, 0.0], top_k=5, min_score=0.0)
    assert result == []


def test_chroma_add_upserts_duplicate_ids() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_upsert",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Original text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Updated text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=5, min_score=0.1)

    assert len(result) == 1
    assert result[0].text == "Updated text"


def test_chroma_query_result_has_score() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_query_score",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Some text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert isinstance(result[0].score, float)


def test_chroma_ephemeral_constructor_requires_no_args() -> None:
    store = ChromaVectorStore()

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Hello",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert len(result) == 1


def test_chroma_persistent_constructor(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        collection_name="test_persistent",
        path=str(tmp_path),
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Persisted text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert len(result) == 1

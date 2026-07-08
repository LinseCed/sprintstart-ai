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


def test_chroma_round_trips_source_role() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test_source_role", client=client)

    store.add(
        [
            Chunk(
                id="chunk-test",
                artifact_id="artifact-1",
                filename="test_foo.py",
                text="test material",
                embedding=[1.0, 0.0],
                source_role="test",
            ),
        ]
    )

    scored = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)
    assert scored[0].source_role == "test"

    listed = store.all_chunks()
    assert listed[0].source_role == "test"


def test_chroma_legacy_chunks_default_to_primary() -> None:
    """A chunk stored without a source_role reads back as 'primary'."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="legacy", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["legacy-1"],
        documents=["legacy text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"artifact_id": "a", "filename": "doc.md", "kind": "text"}],
    )
    store = ChromaVectorStore(collection_name="legacy", client=client)

    scored = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert scored[0].source_role == "primary"


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


def test_chroma_round_trips_start_line_and_start_page() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test_start_line_page", client=client)

    store.add(
        [
            Chunk(
                id="chunk-code",
                artifact_id="artifact-1",
                filename="foo.py",
                text="def foo(): pass",
                embedding=[1.0, 0.0],
                start_line=12,
            ),
            Chunk(
                id="chunk-pdf",
                artifact_id="artifact-1",
                filename="doc.pdf",
                text="PDF text",
                embedding=[0.0, 1.0],
                start_page=3,
            ),
        ]
    )

    scored_code = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)
    assert scored_code[0].start_line == 12
    assert scored_code[0].start_page is None

    scored_pdf = store.query(embedding=[0.0, 1.0], top_k=1, min_score=0.1)
    assert scored_pdf[0].start_page == 3
    assert scored_pdf[0].start_line is None

    listed = store.all_chunks()
    by_id = {chunk.id: chunk for chunk in listed}
    assert by_id["chunk-code"].start_line == 12
    assert by_id["chunk-pdf"].start_page == 3


def test_chroma_legacy_chunks_without_line_or_page_default_to_none() -> None:
    """A chunk stored before start_line/start_page existed reads back as None."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="legacy_line_page", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["legacy-1"],
        documents=["legacy text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"artifact_id": "a", "filename": "doc.md", "kind": "text"}],
    )
    store = ChromaVectorStore(collection_name="legacy_line_page", client=client)

    scored = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert scored[0].start_line is None
    assert scored[0].start_page is None

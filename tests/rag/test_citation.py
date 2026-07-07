from rag.citation import build_citations
from rag.types import ScoredChunk


def test_build_citations_maps_chunk_fields() -> None:
    chunks = [
        ScoredChunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="architecture.md",
            text="Some text",
            score=0.9,
        )
    ]

    citations = build_citations(chunks)

    assert len(citations) == 1
    assert citations[0].filename == "architecture.md"
    assert citations[0].chunk_id == "chunk-1"
    assert citations[0].artifact_id == "artifact-1"

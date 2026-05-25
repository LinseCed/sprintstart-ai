from src.rag.citation import build_citations
from src.rag.types import Chunk


def test_build_citations_maps_heading_path_to_section_path() -> None:
    chunks = [
        Chunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="architecture.md",
            heading_path="Intro > Architecture > Components",
            text="Some text",
            embedding=[1.0, 0.0],
        )
    ]

    citations = build_citations(chunks)

    assert len(citations) == 1
    assert citations[0].filename == "architecture.md"
    assert citations[0].section_path == "Intro > Architecture > Components"
    assert citations[0].chunk_id == "chunk-1"

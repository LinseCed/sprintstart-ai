from rag.citation import build_citations
from rag.types import ScoredChunk


def test_build_citations_maps_heading_path_to_section_path() -> None:
    chunks = [
        ScoredChunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="architecture.md",
            heading_path="Intro > Architecture > Components",
            text="Some text",
            score=0.9,
        )
    ]

    citations = build_citations(chunks)

    assert len(citations) == 1
    assert citations[0].filename == "architecture.md"
    assert citations[0].section_path == "Intro > Architecture > Components"
    assert citations[0].chunk_id == "chunk-1"


def test_build_citations_section_path_is_none_when_no_heading() -> None:
    chunks = [
        ScoredChunk(
            id="chunk-2",
            artifact_id="artifact-1",
            filename="notes.md",
            heading_path=None,
            text="Unstructured text",
            score=0.85,
        )
    ]

    citations = build_citations(chunks)

    assert len(citations) == 1
    assert citations[0].section_path is None
    assert citations[0].filename == "notes.md"
    assert citations[0].chunk_id == "chunk-2"

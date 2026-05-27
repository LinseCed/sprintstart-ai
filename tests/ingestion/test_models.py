from pathlib import Path

import pytest

from ingestion.models import ParsedChunk


@pytest.fixture
def sample_parsed_chunk() -> ParsedChunk:
    path = Path("helloWorld.txt")

    return ParsedChunk(
        content="Hello World!",
        kind="text",
        metadata={
            "filename": path.name,
            "type": path.suffix,
            "source": str(path.resolve()),
        },
    )


def test_get_chunk_content(sample_parsed_chunk: ParsedChunk):
    assert sample_parsed_chunk.content == "Hello World!"


def test_get_chunk_kind(sample_parsed_chunk: ParsedChunk):
    assert sample_parsed_chunk.kind == "text"


def test_get_chunk_metadata(sample_parsed_chunk: ParsedChunk):
    assert sample_parsed_chunk.metadata == {
        "filename": "helloWorld.txt",
        "type": ".txt",
        "source": str(Path("helloWorld.txt").resolve()),
    }

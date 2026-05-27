from pathlib import Path

import pytest

from src.ingestion.models import ParsedChunk
from src.ingestion.parser import parse

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def markdown_file_content() -> bytes:
    return (FIXTURES_DIR/"markdown_sample.md").read_bytes()

@pytest.fixture
def text_file_content() -> bytes:
    return (FIXTURES_DIR/"text_sample.txt").read_bytes()

@pytest.fixture
def json_file_content() -> bytes:
    return (FIXTURES_DIR/"json_sample.json").read_bytes()

def test_parse_markdown_as_single_chunk(markdown_file_content: bytes):
    filename = "markdown_sample.md"
    result: list[ParsedChunk] = parse(filename, markdown_file_content)

    assert len(result) == 1

    chunk = result[0]
    
    assert chunk.kind == "text"
    assert chunk.content == markdown_file_content.decode("utf-8", errors="replace")

    metadata = chunk.metadata
    assert metadata["filename"] == filename
    assert metadata["type"] == ".md"
    assert metadata["source"].endswith(filename)

def test_parse_txt_file_as_single_chunk(text_file_content: bytes):
    filename = "text_sample.txt"
    result: list[ParsedChunk] = parse(filename, text_file_content)

    assert len(result) == 1

    chunk = result[0]
    
    assert chunk.kind == "text"
    assert chunk.content == text_file_content.decode("utf-8", errors="replace")

    metadata = chunk.metadata
    assert metadata["filename"] == filename
    assert metadata["type"] == ".txt"
    assert metadata["source"].endswith(filename)

def test_parse_json_as_single_chunk(json_file_content: bytes):
    filename = "json_sample.json"
    result: list[ParsedChunk] = parse(filename, json_file_content)

    assert len(result) == 1

    chunk = result[0]
    
    assert chunk.kind == "text"
    assert chunk.content == json_file_content.decode("utf-8", errors="replace")

    metadata = chunk.metadata
    assert metadata["filename"] == filename
    assert metadata["type"] == ".json"
    assert metadata["source"].endswith(filename)


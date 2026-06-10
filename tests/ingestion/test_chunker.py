import importlib
import sys

import pytest
from pytest import MonkeyPatch

from ingestion.chunker import chunk_text
from ingestion.code_parser import chunk_code
from ingestion.parser import parse


def test_chunk_overlap_must_be_smaller_than_chunk_size(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    sys.modules.pop("ingestion.chunker", None)

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        importlib.import_module("ingestion.chunker")


def test_chunk_overlap_larger_than_chunk_size_raises(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")

    sys.modules.pop("ingestion.chunker", None)

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        importlib.import_module("ingestion.chunker")


def test_valid_chunk_configuration(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("CHUNK_OVERLAP", "64")

    sys.modules.pop("ingestion.chunker", None)

    module = importlib.import_module("ingestion.chunker")

    assert module.chunk_size == 512
    assert module.chunk_overlap == 64


def test_chunks_have_correct_order():
    filename = "big.txt"
    content = b"A" * 1500

    result = parse(filename, content)

    for i, chunk in enumerate(result):
        assert chunk.metadata["chunk_index"] == str(i)


def test_large_paragraph_is_hard_split():
    text = "A" * 1200

    chunks = chunk_text(
        "file.txt",
        text,
        chunk_size=512,
        chunk_overlap=64,
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.content) <= 512


def test_chunk_code_respects_size_directly():
    code = "\n".join(
        [
            "def foo():",
            "    pass",
        ]
        * 200
    )

    chunks = chunk_code("test.py", code, chunk_size=50)

    for chunk in chunks[:-1]:
        assert len(chunk.content) <= 50


def test_text_chunks_split_on_paragraph_boundaries():
    text = "Paragraph A\n\nParagraph B\n\nParagraph C"

    chunks = chunk_text("file.txt", text, chunk_size=25)

    assert len(chunks) == 2

    assert "Paragraph A" in chunks[0].content
    assert "Paragraph B" in chunks[0].content

    assert "Paragraph B" in chunks[1].content
    assert "Paragraph C" in chunks[1].content


def test_chunking_does_not_cut_typical_prose_mid_sentence():
    text = (
        "This is the first paragraph. "
        "It contains several complete sentences.\n\n"
        "This is the second paragraph. "
        "It also contains complete sentences.\n\n"
        "This is the third paragraph."
    )

    chunks = chunk_text(
        "file.txt",
        text,
        chunk_size=120,
    )

    for chunk in chunks:
        assert not chunk.content.startswith("t contains")
        assert not chunk.content.startswith("plete sentences")


def test_last_paragraph_is_used_as_overlap():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = chunk_text(
        "file.txt",
        text,
        chunk_size=35,
    )

    assert len(chunks) == 2

    assert "Second paragraph." in chunks[0].content
    assert chunks[1].content.startswith("Second paragraph.")

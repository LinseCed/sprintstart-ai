from ingestion.chunker import chunk_text
from ingestion.code_parser import chunk_code
from ingestion.parser import parse


def test_chunks_have_correct_order():
    filename = "big.txt"
    content = b"A" * 1500

    result = parse(filename, content)

    for i, chunk in enumerate(result):
        assert chunk.metadata["chunk_index"] == str(i)


def test_chunk_text_splits_correctly():
    text = "A" * 1200
    chunks = chunk_text("file.txt", text)

    assert len(chunks) == 3


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

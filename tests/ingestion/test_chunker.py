from ingestion.chunker import chunk_text
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

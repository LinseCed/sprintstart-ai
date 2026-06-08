from pathlib import Path

import pytest

from ingestion.pdf_parser import parse_pdf

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def pdf_blank_page() -> bytes:
    return (FIXTURES_DIR / "blank_page.pdf").read_bytes()


@pytest.fixture
def pdf_single_page() -> bytes:
    return (FIXTURES_DIR / "single_page.pdf").read_bytes()


@pytest.fixture
def pdf_multi_page() -> bytes:
    return (FIXTURES_DIR / "multi_page.pdf").read_bytes()


@pytest.fixture
def pdf_long_page() -> bytes:
    return (FIXTURES_DIR / "long_page.pdf").read_bytes()


def test_single_page_pdf(pdf_single_page: bytes):

    result = parse_pdf("single_page.pdf", pdf_single_page)

    assert len(result) > 0
    assert all(chunk.kind == "pdf" for chunk in result)
    assert all(chunk.metadata["page_number"] == "1" for chunk in result)
    assert result[0].metadata["global_pdf_chunk_index"] == "0"


def test_multi_page_pdf(pdf_multi_page: bytes):

    result = parse_pdf("multi_page.pdf", pdf_multi_page)

    assert len(result) > 0
    pages = {chunk.metadata["page_number"] for chunk in result}
    assert "1" in pages
    assert "2" in pages
    assert "3" in pages
    assert all(chunk.kind == "pdf" for chunk in result)
    indices = [int(chunk.metadata["global_pdf_chunk_index"]) for chunk in result]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))


def test_pdf_with_blank_page(pdf_blank_page: bytes):

    result = parse_pdf("blank_page.pdf", pdf_blank_page)

    pages = {chunk.metadata["page_number"] for chunk in result}
    assert len(result) == 0
    assert "1" not in pages

    assert all(chunk.content.strip() != "" for chunk in result)


def test_pdf_parser_basic_properties(pdf_multi_page: bytes):

    result = parse_pdf("multi_page.pdf", pdf_multi_page)

    assert all(chunk.content for chunk in result)
    for chunk in result:
        assert "page_number" in chunk.metadata
        assert "global_pdf_chunk_index" in chunk.metadata
        assert chunk.kind == "pdf"


def test_pdf_long_page_splits_into_multiple_chunks(pdf_long_page: bytes):

    result = parse_pdf("long_page.pdf", pdf_long_page)
    assert len(result) > 1
    assert all(chunk.kind == "pdf" for chunk in result)
    assert all(chunk.metadata["page_number"] == "1" for chunk in result)
    assert all(len(chunk.content) <= 512 for chunk in result)
    indices = [int(chunk.metadata["global_pdf_chunk_index"]) for chunk in result]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))
    assert min(indices) == 0

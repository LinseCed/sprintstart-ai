from pathlib import Path

from ingestion.utils import build_metadata


def test_build_metadata_returns_correct_metadata():
    path = Path("example.txt")

    result = build_metadata(path)

    assert result["filename"] == "example.txt"
    assert result["type"] == ".txt"
    assert result["source"].endswith("example.txt")


def test_build_metadata_returns_all_expected_keys():
    path = Path("document.md")

    result = build_metadata(path)

    assert set(result.keys()) == {"source", "filename", "type"}


def test_build_metadata_values_are_strings():
    path = Path("config.json")

    result = build_metadata(path)

    assert isinstance(result["source"], str)
    assert isinstance(result["filename"], str)
    assert isinstance(result["type"], str)

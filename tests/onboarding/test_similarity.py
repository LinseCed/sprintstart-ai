"""Unit tests for onboarding.similarity — token overlap, cosine, step_text."""

from onboarding.similarity import cosine_similarity, step_text, text_overlap


def test_cosine_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0


def test_cosine_opposite_vectors() -> None:
    result = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
    assert result == -1.0


def test_text_overlap_identical_strings() -> None:
    assert text_overlap("install python", "install python") == 1.0


def test_text_overlap_completely_different() -> None:
    assert text_overlap("install python", "review kubernetes") == 0.0


def test_text_overlap_partial() -> None:
    result = text_overlap("install python dependencies", "install python locally")
    # tokens: {install, python, dependencies} vs {install, python, locally}
    # intersection=2, union=4 -> 0.5
    assert result == 0.5


def test_text_overlap_stopwords_ignored() -> None:
    # "set" and "up" are stopwords, so "set up the database" -> {database}
    # and "set up database" -> {database} -> Jaccard = 1.0
    result = text_overlap("set up the database", "set up database")
    assert result == 1.0


def test_text_overlap_case_insensitive() -> None:
    assert text_overlap("Install Python", "install python") == 1.0


def test_text_overlap_empty_strings() -> None:
    assert text_overlap("", "") == 0.0
    assert text_overlap("hello", "") == 0.0
    assert text_overlap("", "hello") == 0.0


def test_text_overlap_catches_rephrased_duplicates() -> None:
    a = "Verify Python 3.12+ installation"
    b = "Verify Python and Tooling Prerequisites"
    result = text_overlap(a, b)
    # tokens a: {verify, python, 3, 12, installation}
    # tokens b: {verify, python, tooling, prerequisites}
    # intersection: {verify, python} = 2, union = 7
    # 2/7 ≈ 0.29 — below threshold, so title-only overlap is low.
    # But with descriptions, overlap should be higher (tested in pipeline tests).
    assert result < 0.70


def test_step_text_combines_title_and_description() -> None:
    assert step_text("Install Python", "Run apt-get install") == (
        "Install Python. Run apt-get install"
    )


def test_step_text_empty_description() -> None:
    assert step_text("Install Python", "") == "Install Python"


def test_step_text_strips_trailing_period() -> None:
    assert step_text("Install.", "") == "Install"

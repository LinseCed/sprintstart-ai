"""The evidence helpers each assembly pipeline carries its own copy of.

Orientation, phase and diagram assembly share these two helpers by copy, so one
parametrized test holds all three to the same behaviour.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

from onboarding import diagram, orientation, phase
from rag.types import ScoredChunk

_PIPELINES = pytest.mark.parametrize(
    "pipeline", [orientation, phase, diagram], ids=lambda m: m.__name__
)


def _scored(
    chunk_id: str,
    text: str,
    score: float,
    artifact_type: str | None = None,
    language: str | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        id=chunk_id,
        artifact_id=f"a-{chunk_id}",
        filename=f"{chunk_id}.md",
        text=text,
        score=score,
        artifact_type=artifact_type,
        language=language,
    )


@_PIPELINES
def test_duplicates_collapse_into_the_best_scoring_copy(pipeline: ModuleType) -> None:
    collapse: Callable[[list[ScoredChunk]], tuple[list[ScoredChunk], int]]
    collapse = pipeline._collapse_duplicates
    text = "run make dev to start the reports service locally on port 8080"
    chunks = [
        _scored("weak", text, 0.4),
        _scored("best", text, 0.9),
        _scored("middle", text + " .", 0.6),
        _scored("other", "pull requests need one approving review", 0.5),
    ]

    kept, collapsed = collapse(chunks)

    assert [c.id for c in kept] == ["best", "other"]
    assert collapsed == 2


@_PIPELINES
def test_an_evidence_line_names_its_file_and_kind(pipeline: ModuleType) -> None:
    line: Callable[[ScoredChunk], str] = pipeline._evidence_line
    plain = _scored("c1", "run make dev", 0.9)
    code = _scored("c2", "fun main()", 0.9, artifact_type="CODE", language="kotlin")

    assert line(plain) == "  [c1] (c1.md | FILE) run make dev"
    assert line(code) == "  [c2] (c2.md | CODE/kotlin) fun main()"

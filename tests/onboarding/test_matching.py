from onboarding.matching import HireCompetency, match_hire_to_pool
from onboarding.starter_work import ProposedStarterTask
from tests.stubs.llm import StubLLMClient


def _task(
    source_id: str,
    keys: list[str],
    title: str = "Task",
    summary: str = "",
) -> ProposedStarterTask:
    return ProposedStarterTask(
        source_id=source_id, title=title, summary=summary, competency_keys=keys
    )


def test_full_overlap_ranks_above_partial_and_none() -> None:
    hire = [
        HireCompetency(key="kotlin", label="Kotlin"),
        HireCompetency(key="spring", label="Spring"),
    ]
    full = _task("full", ["kotlin", "spring"])
    partial = _task("partial", ["kotlin"])
    none = _task("none", ["react"])
    llm = StubLLMClient(embedding=[0.0] * 768)

    ranked = match_hire_to_pool(llm, hire, [none, partial, full])

    assert [r.task.source_id for r in ranked] == ["full", "partial", "none"]


def test_matched_competency_keys_are_the_intersection() -> None:
    hire = [HireCompetency(key="kotlin", label="Kotlin")]
    task = _task("t1", ["kotlin", "react"])
    llm = StubLLMClient(embedding=[0.0] * 768)

    [ranked] = match_hire_to_pool(llm, hire, [task])

    assert ranked.matched_competency_keys == ["kotlin"]


def test_zero_overlap_broken_by_embedding_similarity() -> None:
    hire = [HireCompetency(key="kotlin", label="Kotlin", description="JVM language")]
    closer = _task("closer", [], title="Kotlin coroutines cleanup")
    farther = _task("farther", [], title="Update marketing copy")

    def embed_fn(text: str) -> list[float]:
        return [1.0, 0.0] if "Kotlin" in text else [0.0, 1.0]

    llm = StubLLMClient(embed_fn=embed_fn)

    ranked = match_hire_to_pool(llm, hire, [farther, closer])

    assert [r.task.source_id for r in ranked] == ["closer", "farther"]


def test_key_overlap_always_outranks_zero_overlap_regardless_of_similarity() -> None:
    hire = [HireCompetency(key="kotlin", label="Kotlin", description="JVM language")]
    # Topically identical to the hire's competency text but untagged.
    untagged_but_similar = _task("untagged", [], title="Kotlin", summary="JVM language")
    tagged_but_dissimilar = _task(
        "tagged", ["kotlin"], title="Update marketing copy", summary=""
    )
    llm = StubLLMClient(embedding=[1.0, 0.0])

    ranked = match_hire_to_pool(
        llm, hire, [untagged_but_similar, tagged_but_dissimilar]
    )

    assert [r.task.source_id for r in ranked] == ["tagged", "untagged"]


def test_empty_pool_returns_empty_list() -> None:
    llm = StubLLMClient()

    ranked = match_hire_to_pool(llm, [HireCompetency(key="k", label="K")], [])

    assert ranked == []

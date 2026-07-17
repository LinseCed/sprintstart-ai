import json

from onboarding.generation import corpus_fingerprint
from onboarding.lessons import synthesize_lesson
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767


def _llm(title: str, body: str, chunk_ids: list[str]) -> StubLLMClient:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {"title": title, "body": body, "chunk_ids": chunk_ids}
        )
    )
    llm.embedding = _EMBED
    return llm


def _store(*texts: str) -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id="a1",
                filename=f"doc{i}.kt",
                text=text,
                embedding=_EMBED,
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store


def test_first_time_synthesis_produces_grounded_lesson() -> None:
    store = _store("Kotlin is the primary backend language for this service")
    llm = _llm("Kotlin basics", "Kotlin is the primary backend language.[c1]", ["c1"])

    outcome = synthesize_lesson(
        llm,
        store,
        competency_key="kotlin",
        competency_label="Kotlin",
        competency_description="Primary backend language",
    )

    assert outcome.status == "synthesized"
    assert outcome.lesson is not None
    assert outcome.lesson.competency_key == "kotlin"
    assert outcome.lesson.level == "beginner"
    assert outcome.lesson.citations[0].chunk_id == "c1"
    assert outcome.provenance is not None
    assert outcome.provenance.corpus_fingerprint == corpus_fingerprint(store)


def test_unchanged_fingerprint_skips_regeneration() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm("Kotlin basics", "body[c1]", ["c1"])
    fingerprint = corpus_fingerprint(store)

    outcome = synthesize_lesson(
        llm,
        store,
        competency_key="kotlin",
        competency_label="Kotlin",
        last_fingerprint=fingerprint,
    )

    assert outcome.status == "unchanged"
    assert outcome.lesson is None


def test_ungrounded_lesson_is_skipped() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm("Kotlin basics", "invented body", ["nonexistent"])

    outcome = synthesize_lesson(
        llm, store, competency_key="kotlin", competency_label="Kotlin"
    )

    assert outcome.status == "skipped"
    assert outcome.lesson is None


def test_empty_corpus_is_skipped() -> None:
    store = StubVectorStore()
    llm = _llm("Kotlin basics", "body", [])

    outcome = synthesize_lesson(
        llm, store, competency_key="kotlin", competency_label="Kotlin"
    )

    assert outcome.status == "skipped"


def test_malformed_llm_output_is_skipped_not_raised() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = StubLLMClient(generate_response="not json")
    llm.embedding = _EMBED

    outcome = synthesize_lesson(
        llm, store, competency_key="kotlin", competency_label="Kotlin"
    )

    assert outcome.status == "skipped"
    assert outcome.lesson is None


def test_level_is_carried_onto_the_lesson() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm("Kotlin basics", "body[c1]", ["c1"])

    outcome = synthesize_lesson(
        llm,
        store,
        competency_key="kotlin",
        competency_label="Kotlin",
        level="advanced",
    )

    assert outcome.lesson is not None
    assert outcome.lesson.level == "advanced"

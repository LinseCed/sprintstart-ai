import json

from onboarding.generation import corpus_fingerprint
from onboarding.graph_generation import generate_competency_graph
from onboarding.graph_models import ActiveCompetency, ActiveEdge
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_OTHER_EMBED = [0.0, 1.0] + [0.0] * 766


def _llm(
    competencies: list[dict[str, object]], edges: list[dict[str, object]] | None = None
) -> StubLLMClient:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {"competencies": competencies, "edges": edges or []}
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


def test_first_time_proposal_drafts_grounded_competencies() -> None:
    store = _store("Kotlin is the primary backend language for this service")
    llm = _llm(
        [
            {
                "key": "kotlin",
                "label": "Kotlin",
                "description": "Primary backend language",
                "kind": "SKILL",
                "repo_ref": "build.gradle.kts",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcome = generate_competency_graph(llm, store)

    assert outcome.status == "proposed"
    assert [c.key for c in outcome.competencies] == ["kotlin"]
    assert outcome.competencies[0].citations[0].chunk_id == "c1"
    assert outcome.provenance is not None
    assert outcome.provenance.corpus_fingerprint == corpus_fingerprint(store)


def test_ungrounded_competency_is_dropped() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [
            {
                "key": "invented",
                "label": "Invented Thing",
                "kind": "SKILL",
                "chunk_ids": ["nonexistent"],
            }
        ]
    )

    outcome = generate_competency_graph(llm, store)

    assert outcome.status == "skipped"
    assert outcome.competencies == []


def test_edges_reference_proposed_and_active_competencies() -> None:
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [
            {
                "key": "kotlin",
                "label": "Kotlin",
                "kind": "SKILL",
                "chunk_ids": ["c1"],
            }
        ],
        edges=[
            {
                "from_key": "kotlin",
                "to_key": "our-domain-model",
                "rationale": "Domain model is written in Kotlin",
            }
        ],
    )
    # Distinct embeddings per label so the dedup gate doesn't mistake "Kotlin" for
    # the unrelated active "Our Domain Model" -- StubLLMClient otherwise returns
    # the same fixed embedding for every text. Exact match (not substring), since
    # the retrieval query text itself also happens to contain "domain".
    llm.embed_fn = lambda text: _OTHER_EMBED if text == "Our Domain Model" else _EMBED
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    outcome = generate_competency_graph(llm, store, active_competencies=active)

    assert len(outcome.edges) == 1
    edge = outcome.edges[0]
    assert edge.from_key == "kotlin"
    assert edge.to_key == "our-domain-model"
    assert edge.rationale == "Domain model is written in Kotlin"


def test_edge_with_self_loop_is_dropped() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[{"from_key": "kotlin", "to_key": "kotlin", "rationale": "n/a"}],
    )

    outcome = generate_competency_graph(llm, store)

    assert outcome.edges == []


def test_edge_to_unknown_key_is_dropped() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[{"from_key": "kotlin", "to_key": "does-not-exist", "rationale": "n/a"}],
    )

    outcome = generate_competency_graph(llm, store)

    assert outcome.edges == []


def test_edge_already_active_is_not_reproposed() -> None:
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[
            {"from_key": "kotlin", "to_key": "our-domain-model", "rationale": "n/a"}
        ],
    )
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]
    active_edges = [ActiveEdge(from_key="kotlin", to_key="our-domain-model")]

    outcome = generate_competency_graph(
        llm, store, active_competencies=active, active_edges=active_edges
    )

    assert outcome.edges == []


def test_active_competency_is_never_reproposed() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}]
    )
    active = [ActiveCompetency(key="kotlin", label="Kotlin", kind="SKILL")]

    outcome = generate_competency_graph(llm, store, active_competencies=active)

    assert outcome.status == "skipped"
    assert outcome.competencies == []


def test_near_duplicate_competency_is_dropped_via_embedding_similarity() -> None:
    store = _store("Kotlin is the primary backend language and Kotlin lang basics")
    llm = _llm(
        [
            {"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]},
            {
                "key": "kotlin-lang",
                "label": "Kotlin Language",
                "kind": "SKILL",
                "chunk_ids": ["c1"],
            },
        ]
    )
    # Both proposals embed identically (StubLLMClient returns a fixed embedding),
    # so the second must be dropped as a near-duplicate of the first.

    outcome = generate_competency_graph(llm, store)

    assert [c.key for c in outcome.competencies] == ["kotlin"]


def test_key_is_normalized_to_kebab_case() -> None:
    store = _store("Spring Boot powers the web layer")
    llm = _llm(
        [
            {
                "key": "Spring Boot!",
                "label": "Spring Boot",
                "kind": "SKILL",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcome = generate_competency_graph(llm, store)

    assert outcome.competencies[0].key == "spring-boot"


def test_unchanged_corpus_with_matching_fingerprint_is_a_noop() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}]
    )

    outcome = generate_competency_graph(
        llm, store, last_fingerprint=corpus_fingerprint(store)
    )

    assert outcome.status == "unchanged"
    assert outcome.competencies == []


def test_empty_corpus_is_skipped() -> None:
    store = StubVectorStore()
    llm = _llm([])

    outcome = generate_competency_graph(llm, store)

    assert outcome.status == "skipped"


def test_invalid_llm_json_is_skipped_not_raised() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = StubLLMClient(generate_response="not json at all")
    llm.embedding = _EMBED

    outcome = generate_competency_graph(llm, store)

    assert outcome.status == "skipped"


def test_reruns_on_the_same_corpus_propose_the_same_graph() -> None:
    """Stability across reruns: an unchanged corpus and deterministic LLM output
    (as a fixed-fixture LLM is, in these tests) reproposes the same graph."""
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[
            {"from_key": "kotlin", "to_key": "our-domain-model", "rationale": "n/a"}
        ],
    )
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    first = generate_competency_graph(llm, store, active_competencies=active)
    second = generate_competency_graph(llm, store, active_competencies=active)

    assert first.competencies == second.competencies
    assert first.edges == second.edges

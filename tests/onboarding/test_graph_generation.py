import json
from collections.abc import Generator

from llm.base import Message
from onboarding.generation import corpus_fingerprint
from onboarding.graph_generation import (
    generate_competency_graph,
    stream_competency_graph,
)
from onboarding.graph_models import ActiveCompetency, ActiveEdge
from onboarding.progress import ProgressEvent
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def _collect[T](
    generator: Generator[ProgressEvent, None, T],
) -> tuple[list[ProgressEvent], T]:
    """Drain a progress generator, keeping both the events and the returned value."""
    events: list[ProgressEvent] = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        return events, stop.value


_EMBED = [1.0] + [0.0] * 767
_OTHER_EMBED = [0.0, 1.0] + [0.0] * 766


class _TwoPassLLM(StubLLMClient):
    """Node generation and edge generation are separate calls, so the stub has to
    answer them separately: the node payload first, the edge payload after."""

    def __init__(
        self,
        node_response: str,
        edge_response: str,
        embedding: list[float] | None = None,
    ) -> None:
        super().__init__(generate_response=node_response, embedding=embedding)
        self.node_response = node_response
        self.edge_response = edge_response
        self.prompts: list[list[Message]] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.prompts.append(messages)
        is_edge_pass = "relationships" in str(messages[0]["content"])
        return self.edge_response if is_edge_pass else self.node_response


def _llm(
    competencies: list[dict[str, object]],
    edges: list[dict[str, object]] | None = None,
    *,
    isolated: list[dict[str, object]] | None = None,
) -> _TwoPassLLM:
    llm = _TwoPassLLM(
        node_response=json.dumps({"competencies": competencies}),
        edge_response=json.dumps(
            {"tiers": [], "edges": edges or [], "isolated": isolated or []}
        ),
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
    llm = _TwoPassLLM(node_response="not json at all", edge_response="not json either")
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


def test_edge_pass_runs_separately_from_the_node_pass() -> None:
    """Nodes and edges are two calls: a single call spent its budget on nodes and
    left the graph a scatter (10 nodes, 3 edges in live testing)."""
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[
            {
                "from_key": "kotlin",
                "to_key": "our-domain-model",
                "kind": "RELATED",
                "rationale": "n/a",
            }
        ],
    )
    llm.embed_fn = lambda text: _OTHER_EMBED if text == "Our Domain Model" else _EMBED
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    generate_competency_graph(llm, store, active_competencies=active)

    assert len(llm.prompts) == 2
    assert "relationships" in str(llm.prompts[1][0]["content"])


def test_related_edges_are_proposed_and_kept_non_gating() -> None:
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[
            {
                "from_key": "kotlin",
                "to_key": "our-domain-model",
                "kind": "RELATED",
                "rationale": "Both live in the backend service",
            }
        ],
    )
    llm.embed_fn = lambda text: _OTHER_EMBED if text == "Our Domain Model" else _EMBED
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    outcome = generate_competency_graph(llm, store, active_competencies=active)

    assert [e.kind for e in outcome.edges] == ["RELATED"]


def test_unknown_edge_kind_falls_back_to_the_non_gating_one() -> None:
    """An unrecognised kind must not become PREREQUISITE: guessing wrong there
    locks a node a hire has no reason to be blocked on."""
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[
            {
                "from_key": "kotlin",
                "to_key": "our-domain-model",
                "kind": "BUILDS_ON",
                "rationale": "n/a",
            }
        ],
    )
    llm.embed_fn = lambda text: _OTHER_EMBED if text == "Our Domain Model" else _EMBED
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    outcome = generate_competency_graph(llm, store, active_competencies=active)

    assert [e.kind for e in outcome.edges] == ["RELATED"]


def test_prerequisite_edge_that_closes_a_cycle_is_dropped() -> None:
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[
            {
                "from_key": "our-domain-model",
                "to_key": "kotlin",
                "kind": "PREREQUISITE",
                "rationale": "reverses an edge already in the graph",
            }
        ],
    )
    llm.embed_fn = lambda text: _OTHER_EMBED if text == "Our Domain Model" else _EMBED
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]
    active_edges = [
        ActiveEdge(from_key="kotlin", to_key="our-domain-model", kind="PREREQUISITE")
    ]

    outcome = generate_competency_graph(
        llm, store, active_competencies=active, active_edges=active_edges
    )

    assert outcome.edges == []
    assert any("cycle" in note for note in outcome.notes)


def test_related_cycle_is_allowed_because_it_gates_nothing() -> None:
    """Only PREREQUISITE locks a node, so a RELATED edge that closes a loop with
    existing prerequisites is structure, not a deadlock."""
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        edges=[
            {
                "from_key": "testing",
                "to_key": "kotlin",
                "kind": "RELATED",
                "rationale": "n/a",
            }
        ],
    )
    # Exact-match only: the retrieval query text must keep the chunk embedding,
    # or nothing is retrieved and the node pass never runs.
    llm.embed_fn = lambda text: (
        _OTHER_EMBED if text in ("Our Domain Model", "Testing") else _EMBED
    )
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        ),
        ActiveCompetency(key="testing", label="Testing", kind="SKILL"),
    ]
    active_edges = [
        ActiveEdge(from_key="kotlin", to_key="our-domain-model", kind="PREREQUISITE"),
        ActiveEdge(from_key="our-domain-model", to_key="testing", kind="PREREQUISITE"),
    ]

    outcome = generate_competency_graph(
        llm, store, active_competencies=active, active_edges=active_edges
    )

    assert [(e.from_key, e.to_key) for e in outcome.edges] == [("testing", "kotlin")]


def test_unchanged_corpus_still_proposes_missing_edges() -> None:
    """The scatter this fixes lives in an already-generated graph, so an
    unchanged corpus must not short-circuit the relationships pass."""
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [],
        edges=[
            {
                "from_key": "kotlin",
                "to_key": "our-domain-model",
                "kind": "PREREQUISITE",
                "rationale": "The domain model is written in Kotlin",
            }
        ],
    )
    active = [
        ActiveCompetency(key="kotlin", label="Kotlin", kind="SKILL"),
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        ),
    ]

    outcome = generate_competency_graph(
        llm,
        store,
        active_competencies=active,
        last_fingerprint=corpus_fingerprint(store),
    )

    assert outcome.status == "proposed"
    assert outcome.competencies == []
    assert [(e.from_key, e.to_key) for e in outcome.edges] == [
        ("kotlin", "our-domain-model")
    ]


def test_unconnected_nodes_are_reported_to_the_reviewer() -> None:
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}],
        isolated=[{"key": "our-domain-model", "reason": "nothing depends on it yet"}],
    )
    llm.embed_fn = lambda text: _OTHER_EMBED if text == "Our Domain Model" else _EMBED
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    outcome = generate_competency_graph(llm, store, active_competencies=active)

    assert any("nothing depends on it yet" in note for note in outcome.notes)
    assert any("remain unconnected" in note for note in outcome.notes)


def test_failed_edge_pass_still_keeps_the_node_proposals() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _TwoPassLLM(
        node_response=json.dumps(
            {
                "competencies": [
                    {
                        "key": "kotlin",
                        "label": "Kotlin",
                        "kind": "SKILL",
                        "chunk_ids": ["c1"],
                    }
                ]
            }
        ),
        edge_response="not json at all",
    )
    llm.embedding = _EMBED
    llm.embed_fn = lambda text: _OTHER_EMBED if text == "Our Domain Model" else _EMBED
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    outcome = generate_competency_graph(llm, store, active_competencies=active)

    assert outcome.status == "proposed"
    assert [c.key for c in outcome.competencies] == ["kotlin"]
    assert outcome.edges == []


# --- streaming -----------------------------------------------------------------


def test_stream_emits_nodes_then_edges_as_items_and_a_done() -> None:
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [
            {"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]},
            {
                "key": "our-domain-model",
                "label": "Our Domain Model",
                "kind": "CONCEPT",
                "chunk_ids": ["c1"],
            },
        ],
        edges=[
            {
                "from_key": "kotlin",
                "to_key": "our-domain-model",
                "kind": "PREREQUISITE",
                "rationale": "Domain model is written in Kotlin",
            }
        ],
    )

    # Distinct embeddings so the dedup gate doesn't collapse the two new nodes.
    def _embed(text: str) -> list[float]:
        return _OTHER_EMBED if "Domain" in text else _EMBED

    llm.embed_fn = _embed

    events, outcome = _collect(stream_competency_graph(llm, store))

    # Items land nodes first, then edges -- the graph assembles in that order.
    item_types: list[str] = [
        "edge" if "from_key" in e["item"] else "node"  # type: ignore[operator]
        for e in events
        if e["type"] == "item"
    ]
    assert item_types == ["node", "node", "edge"]
    # seq is monotonic across the whole stream.
    assert [e["seq"] for e in events] == list(range(len(events)))
    # The terminal event carries the whole outcome, and it is the returned one.
    assert events[-1]["type"] == "done"
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert outcome.status == "proposed"


def test_stream_never_emits_an_ungrounded_competency_as_an_item() -> None:
    # A competency citing nothing is dropped, so it must never appear as a live
    # item -- that is the whole promise of an `item` event.
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [
            {"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]},
            {
                "key": "invented",
                "label": "Invented",
                "kind": "SKILL",
                "chunk_ids": ["nonexistent"],
            },
        ]
    )

    events, _ = _collect(stream_competency_graph(llm, store))

    node_labels: list[object] = [
        e["item"]["label"]  # type: ignore[index]
        for e in events
        if e["type"] == "item" and "from_key" not in e["item"]  # type: ignore[operator]
    ]
    assert node_labels == ["Kotlin"]


def test_streaming_result_equals_the_non_streaming_proposal() -> None:
    # The stream is a view of the same computation: its final outcome must be what
    # the plain call returns (provenance timestamps aside).
    store = _store("Kotlin is the primary backend language for our domain model")

    def _fresh() -> _TwoPassLLM:
        llm = _llm(
            [
                {
                    "key": "kotlin",
                    "label": "Kotlin",
                    "kind": "SKILL",
                    "chunk_ids": ["c1"],
                }
            ]
        )
        return llm

    _, streamed = _collect(stream_competency_graph(_fresh(), store))
    synchronous = generate_competency_graph(_fresh(), store)

    assert streamed.status == synchronous.status
    assert [c.key for c in streamed.competencies] == [
        c.key for c in synchronous.competencies
    ]
    assert [(e.from_key, e.to_key) for e in streamed.edges] == [
        (e.from_key, e.to_key) for e in synchronous.edges
    ]


def test_an_empty_corpus_streams_a_skipped_done_not_an_error() -> None:
    events, outcome = _collect(stream_competency_graph(_llm([]), StubVectorStore()))

    assert outcome.status == "skipped"
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["status"] == "skipped"  # type: ignore[index]
    assert "error" not in [e["type"] for e in events]

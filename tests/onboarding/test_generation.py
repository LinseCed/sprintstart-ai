import json

from llm.base import Message
from onboarding.generation import corpus_fingerprint, generate_blueprints
from onboarding.graph_models import ActiveCompetency
from onboarding.models import Baseline, BaselineCompetency
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

# Non-zero embedding so the stub store returns a perfect cosine match.
_EMBED = [1.0] + [0.0] * 767
_SCOPE = "area:backend"

_CATALOG = [
    ActiveCompetency(
        key="deploy-runbook",
        label="Deploy the service",
        description="Follow the deploy runbook end to end.",
        kind="SKILL",
    ),
    ActiveCompetency(
        key="rag-pipeline",
        label="RAG pipeline",
        description="How retrieval augmented generation works here.",
        kind="CONCEPT",
    ),
]


def _llm(competencies: list[dict[str, object]]) -> StubLLMClient:
    llm = StubLLMClient(generate_response=json.dumps({"competencies": competencies}))
    llm.embedding = _EMBED
    return llm


def _store(*texts: str) -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id="a1",
                filename=f"doc{i}.md",
                text=text,
                embedding=_EMBED,
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store


def _active(*entries: BaselineCompetency, version: str = "1") -> Baseline:
    """An authored active baseline as the backend would pass it in."""
    return Baseline(
        scope=_SCOPE, version=version, source="authored", competencies=list(entries)
    )


def _selection(key: str, **kwargs: object) -> dict[str, object]:
    return {"competency_key": key, "chunk_ids": ["c1"], **kwargs}


def test_first_time_generation_selects_grounded_competencies() -> None:
    store = _store("backend onboarding deploy runbook local db setup")
    llm = _llm([_selection("deploy-runbook", requirement="required")])

    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE], competencies=_CATALOG)

    assert [(o.scope, o.status) for o in outcomes] == [(_SCOPE, "created")]
    baseline = outcomes[0].blueprint
    assert baseline is not None
    assert baseline.source == "generated"
    assert baseline.version == "1"
    assert [c.competency_key for c in baseline.competencies] == ["deploy-runbook"]
    assert baseline.competencies[0].requirement == "required"
    assert baseline.provenance is not None
    assert baseline.provenance.corpus_fingerprint == corpus_fingerprint(store)


def test_unchanged_corpus_is_a_noop() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm([_selection("deploy-runbook")])

    first = generate_blueprints(llm, store, scopes=[_SCOPE], competencies=_CATALOG)
    # The backend persists the result and passes it back as the active baseline.
    again = generate_blueprints(
        llm,
        store,
        scopes=[_SCOPE],
        active=[first[0].blueprint],  # type: ignore[list-item]
        competencies=_CATALOG,
    )

    assert again[0].status == "unchanged"


def test_corpus_change_updates_active_with_new_version() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm([_selection("deploy-runbook")])

    first = generate_blueprints(llm, store, scopes=[_SCOPE], competencies=_CATALOG)
    active = first[0].blueprint
    assert active is not None

    store.add(
        [
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="d2.md",
                text="new",
                embedding=_EMBED,
            )
        ]
    )
    outcomes = generate_blueprints(
        llm, store, scopes=[_SCOPE], active=[active], competencies=_CATALOG
    )

    assert outcomes[0].status == "updated"
    assert outcomes[0].draft_version == "2"


def test_required_removal_is_blocked_and_reinjected() -> None:
    active = _active(
        BaselineCompetency(competency_key="rag-pipeline", requirement="required"),
    )
    store = _store("backend onboarding deploy runbook")
    # The draft omits the required competency entirely.
    llm = _llm([_selection("deploy-runbook")])

    outcomes = generate_blueprints(
        llm, store, scopes=[_SCOPE], active=[active], competencies=_CATALOG
    )

    assert outcomes[0].status == "escalated"
    baseline = outcomes[0].blueprint
    assert baseline is not None
    keys = {c.competency_key for c in baseline.competencies}
    assert "rag-pipeline" in keys  # protected entry re-injected, never dropped
    assert baseline.provenance is not None
    assert any("rag-pipeline" in note for note in baseline.provenance.notes)


def test_invariant_flag_protects_a_recommended_entry() -> None:
    active = _active(
        BaselineCompetency(
            competency_key="rag-pipeline", requirement="recommended", invariant=True
        ),
    )
    store = _store("backend onboarding")
    llm = _llm([_selection("deploy-runbook")])

    outcomes = generate_blueprints(
        llm, store, scopes=[_SCOPE], active=[active], competencies=_CATALOG
    )

    assert outcomes[0].status == "escalated"
    baseline = outcomes[0].blueprint
    assert baseline is not None
    assert "rag-pipeline" in {c.competency_key for c in baseline.competencies}


def test_lowering_the_bar_of_a_protected_entry_is_restored() -> None:
    active = _active(
        BaselineCompetency(
            competency_key="deploy-runbook", requirement="required", target_level=3
        ),
    )
    store = _store("backend onboarding deploy runbook")
    llm = _llm([_selection("deploy-runbook", requirement="required", target_level=1)])

    outcomes = generate_blueprints(
        llm, store, scopes=[_SCOPE], active=[active], competencies=_CATALOG
    )

    assert outcomes[0].status == "escalated"
    baseline = outcomes[0].blueprint
    assert baseline is not None
    assert baseline.competencies[0].target_level == 3


def test_raising_the_bar_of_a_protected_entry_is_allowed() -> None:
    active = _active(
        BaselineCompetency(
            competency_key="deploy-runbook", requirement="required", target_level=2
        ),
    )
    store = _store("backend onboarding deploy runbook")
    llm = _llm([_selection("deploy-runbook", requirement="required", target_level=4)])

    outcomes = generate_blueprints(
        llm, store, scopes=[_SCOPE], active=[active], competencies=_CATALOG
    )

    baseline = outcomes[0].blueprint
    assert baseline is not None
    assert baseline.competencies[0].target_level == 4
    assert outcomes[0].status == "updated"


def test_ungrounded_selections_are_dropped() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [
            _selection("deploy-runbook"),
            {"competency_key": "rag-pipeline", "chunk_ids": ["missing"]},
        ]
    )

    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE], competencies=_CATALOG)

    baseline = outcomes[0].blueprint
    assert baseline is not None
    assert [c.competency_key for c in baseline.competencies] == ["deploy-runbook"]


def test_repeated_selection_of_the_same_competency_collapses() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm([_selection("deploy-runbook"), _selection("deploy-runbook")])

    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE], competencies=_CATALOG)

    baseline = outcomes[0].blueprint
    assert baseline is not None
    assert [c.competency_key for c in baseline.competencies] == ["deploy-runbook"]


def test_invented_competency_key_is_discarded() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm([_selection("made-up-key"), _selection("deploy-runbook")])

    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE], competencies=_CATALOG)

    baseline = outcomes[0].blueprint
    assert baseline is not None
    assert [c.competency_key for c in baseline.competencies] == ["deploy-runbook"]


def test_target_level_outside_the_rank_range_is_ignored() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm([_selection("deploy-runbook", target_level=9)])

    outcomes = generate_blueprints(llm, store, scopes=[_SCOPE], competencies=_CATALOG)

    baseline = outcomes[0].blueprint
    assert baseline is not None
    assert baseline.competencies[0].target_level is None


def test_empty_catalog_is_skipped_rather_than_proposed_empty() -> None:
    store = _store("backend onboarding deploy runbook")

    outcomes = generate_blueprints(_llm([]), store, scopes=[_SCOPE])

    assert outcomes[0].status == "skipped"
    assert outcomes[0].blueprint is None


def test_empty_corpus_is_skipped() -> None:
    outcomes = generate_blueprints(
        _llm([]), StubVectorStore(), scopes=[_SCOPE], competencies=_CATALOG
    )
    assert outcomes[0].status == "skipped"


def test_area_scope_is_told_what_global_already_requires() -> None:
    """The global selection reaches the area prompt as an exclusion list."""
    store = _store("team onboarding deploy runbook rag pipeline setup")
    llm = _llm([_selection("deploy-runbook")])
    prompts: list[str] = []
    inner = llm.generate

    def recording(messages: list[Message], **kwargs: object) -> str:
        prompts.append(str(messages[0]["content"]))
        return inner(messages)

    llm.generate = recording  # type: ignore[method-assign]

    outcomes = generate_blueprints(
        llm, store, scopes=["global", _SCOPE], competencies=_CATALOG
    )

    assert [o.scope for o in outcomes] == ["global", _SCOPE]
    # The stub returns the same selection for both scopes; what matters is that
    # the area prompt was told which keys global already covers.
    assert "ALREADY IN THE GLOBAL BASELINE" not in prompts[0]
    assert "ALREADY IN THE GLOBAL BASELINE" in prompts[1]

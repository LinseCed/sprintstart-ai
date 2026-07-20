import json

from onboarding.modules import propose_module
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_KEY = "deploy-runbook"
_LABEL = "Deploy the service"


def _llm(payload: dict[str, object]) -> StubLLMClient:
    llm = StubLLMClient(generate_response=json.dumps(payload))
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


def _page(
    kind: str, title: str, chunk_ids: list[str] | None = None
) -> dict[str, object]:
    page: dict[str, object] = {"kind": kind, "title": title, "body": f"{title} body"}
    if chunk_ids is not None:
        page["chunk_ids"] = chunk_ids
    return page


def _payload(*pages: dict[str, object], **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": "Deploying the service",
        "summary": "How deploys work here.",
        "pages": list(pages),
        "verification": {
            "prompt": "Walk through a rollback.",
            "rubric": "Names the runbook step that reverts the release.",
        },
    }
    payload.update(extra)
    return payload


def test_proposes_ordered_typed_pages_and_its_gating_check() -> None:
    store = _store("deploy runbook rollback release process")
    llm = _llm(
        _payload(
            _page("CONTEXT", "Why deploys are gated", ["c1"]),
            _page("LESSON", "How the pipeline works", ["c1"]),
            _page("TASK", "Deploy to staging"),
        )
    )

    outcome = propose_module(
        llm, store, competency_key=_KEY, competency_label=_LABEL, level="beginner"
    )

    assert outcome.status == "proposed"
    module = outcome.module
    assert module is not None
    assert [p.kind for p in module.pages] == ["CONTEXT", "LESSON", "TASK"]
    assert module.competency_key == _KEY
    assert module.pages[0].citations[0].chunk_id == "c1"
    # The gate belongs to the module, not to a per-user step.
    assert module.verification is not None
    assert module.verification.type == "KNOWLEDGE"
    assert module.verification.rubric is not None


def test_drops_an_ungrounded_page_but_keeps_the_module() -> None:
    """One hallucinated page must not ride along on its neighbours' grounding."""
    store = _store("deploy runbook rollback")
    llm = _llm(
        _payload(
            _page("LESSON", "Grounded", ["c1"]),
            _page("WALKTHROUGH", "Invented", ["nope"]),
        )
    )

    outcome = propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)

    module = outcome.module
    assert module is not None
    assert [p.title for p in module.pages] == ["Grounded"]
    assert outcome.pages_dropped == 1


def test_task_and_check_pages_need_no_citations() -> None:
    """They are exercises built on the pages above them, not claims of their own."""
    store = _store("deploy runbook rollback")
    llm = _llm(
        _payload(
            _page("LESSON", "How it works", ["c1"]),
            _page("TASK", "Try a deploy"),
            _page("CHECK", "Quick check"),
        )
    )

    outcome = propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)

    module = outcome.module
    assert module is not None
    assert [p.kind for p in module.pages] == ["LESSON", "TASK", "CHECK"]


def test_drops_a_page_of_a_kind_the_backend_cannot_store() -> None:
    store = _store("deploy runbook rollback")
    llm = _llm(
        _payload(
            _page("LESSON", "How it works", ["c1"]),
            _page("QUIZ", "Not a real kind", ["c1"]),
        )
    )

    outcome = propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)

    module = outcome.module
    assert module is not None
    assert [p.kind for p in module.pages] == ["LESSON"]
    assert outcome.pages_dropped == 1


def test_skips_when_every_page_is_ungrounded() -> None:
    store = _store("deploy runbook rollback")
    llm = _llm(_payload(_page("LESSON", "Invented", ["nope"])))

    outcome = propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)

    assert outcome.status == "skipped"
    assert outcome.module is None


def test_level_changes_the_shape_asked_for() -> None:
    """An expert node is not four pages of basics."""
    store = _store("deploy runbook rollback")
    prompts: list[str] = []

    llm = _llm(_payload(_page("LESSON", "How it works", ["c1"])))
    inner = llm.generate

    def recording(messages: list[dict[str, str]], **kwargs: object) -> str:
        prompts.append(str(messages[0]["content"]))
        return inner(messages)  # type: ignore[arg-type]

    llm.generate = recording  # type: ignore[method-assign]

    propose_module(
        llm, store, competency_key=_KEY, competency_label=_LABEL, level="beginner"
    )
    propose_module(
        llm, store, competency_key=_KEY, competency_label=_LABEL, level="expert"
    )

    assert "first principles" in prompts[0]
    assert "Be terse" in prompts[1]
    assert prompts[0] != prompts[1]


def test_generation_is_sampled_deterministically() -> None:
    """Re-runs must not churn a module a PM has already edited."""
    store = _store("deploy runbook rollback")
    llm = _llm(_payload(_page("LESSON", "How it works", ["c1"])))
    temperatures: list[object] = []
    inner = llm.generate

    def recording(messages: list[dict[str, str]], **kwargs: object) -> str:
        temperatures.append(kwargs.get("temperature"))
        return inner(messages)  # type: ignore[arg-type]

    llm.generate = recording  # type: ignore[method-assign]

    propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)

    assert temperatures == [0.0]


def test_unchanged_corpus_is_a_noop() -> None:
    store = _store("deploy runbook rollback")
    llm = _llm(_payload(_page("LESSON", "How it works", ["c1"])))

    first = propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)
    assert first.provenance is not None
    again = propose_module(
        llm,
        store,
        competency_key=_KEY,
        competency_label=_LABEL,
        last_fingerprint=first.provenance.corpus_fingerprint,
    )

    assert again.status == "unchanged"
    assert again.module is None


def test_same_input_yields_the_same_module() -> None:
    store = _store("deploy runbook rollback")
    llm = _llm(
        _payload(
            _page("CONTEXT", "Why", ["c1"]),
            _page("LESSON", "How", ["c1"]),
        )
    )

    first = propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)
    second = propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)

    assert first.module is not None
    assert second.module is not None
    assert first.module.model_dump() == second.module.model_dump()


def test_a_module_without_a_usable_check_still_proposes_its_pages() -> None:
    """A PM can write the gate; losing the pages over a missing prompt is worse."""
    store = _store("deploy runbook rollback")
    llm = _llm(
        _payload(_page("LESSON", "How it works", ["c1"]), verification={"prompt": ""})
    )

    outcome = propose_module(llm, store, competency_key=_KEY, competency_label=_LABEL)

    assert outcome.status == "proposed"
    assert outcome.module is not None
    assert outcome.module.verification is None


def test_empty_corpus_is_skipped() -> None:
    outcome = propose_module(
        _llm(_payload()),
        StubVectorStore(),
        competency_key=_KEY,
        competency_label=_LABEL,
    )

    assert outcome.status == "skipped"

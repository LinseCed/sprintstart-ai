import json

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from insights.faq import FaqDocument, FaqQuestionInput, group_faqs
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_VPN = [1.0, 0.0, 0.0]
_PASSWORD = [0.0, 1.0, 0.0]


def _embed_fn(text: str) -> list[float]:
    if "VPN" in text:
        return _VPN
    if "password" in text.lower():
        return _PASSWORD
    return [0.0, 0.0, 1.0]


class _EchoLLM(StubLLMClient):
    """Redaction pass-through so tests can focus on clustering/documents."""

    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        payload = json.loads(messages[-1]["content"])  # type: ignore[index]
        return json.dumps({"texts": payload["texts"]})


def _metadata_store() -> IngestionMetadataStore:
    return IngestionMetadataStore(path=":memory:")


def test_group_faqs_clusters_similar_questions_by_embedding() -> None:
    llm = _EchoLLM(embed_fn=_embed_fn)
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [
        FaqQuestionInput(id="q1", text="How do I get VPN access?"),
        FaqQuestionInput(id="q2", text="Can someone enable VPN for me?"),
        FaqQuestionInput(id="q3", text="How do I reset my password?"),
    ]

    groups = group_faqs(questions, llm, store, metadata_store)

    assert [g.count for g in groups] == [2, 1]
    assert groups[0].question == "How do I get VPN access?"
    assert groups[0].questions == [
        "How do I get VPN access?",
        "Can someone enable VPN for me?",
    ]
    assert groups[1].question == "How do I reset my password?"


def test_group_faqs_attaches_documents_from_retrieval() -> None:
    llm = _EchoLLM(embed_fn=_embed_fn)
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc_001",
                filename="vpn-setup.md",
                text="How to get VPN access set up",
                embedding=_VPN,
            )
        ]
    )
    metadata_store = _metadata_store()
    metadata_store.save_completed_artifact(
        ArtifactRecord(
            id="doc_001",
            filename="VPN Setup Guide.md",
            content_type="text/markdown",
            source_type="confluence",
            size_bytes=100,
            chunk_count=1,
            status="completed",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    questions = [FaqQuestionInput(id="q1", text="How do I get VPN access?")]

    groups = group_faqs(questions, llm, store, metadata_store)

    assert len(groups) == 1
    assert groups[0].documents == [
        FaqDocument(id="doc_001", title="VPN Setup Guide.md", source="confluence")
    ]


def test_group_faqs_empty_input_returns_no_groups() -> None:
    llm = _EchoLLM(embed_fn=_embed_fn)
    store = StubVectorStore()
    metadata_store = _metadata_store()

    assert group_faqs([], llm, store, metadata_store) == []


class _RedactingLLM(StubLLMClient):
    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        payload = json.loads(messages[-1]["content"])  # type: ignore[index]
        redacted = [t.replace("John Doe", "[NAME]") for t in payload["texts"]]
        return json.dumps({"texts": redacted})


def test_group_faqs_caps_sample_size_below_total_count() -> None:
    llm = _RedactingLLM(embed_fn=_embed_fn)
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [
        FaqQuestionInput(id=f"q{i}", text=f"How do I get VPN access, {suffix}?")
        for i, suffix in enumerate(["a", "b", "c", "d", "e", "f", "g"])
    ]

    groups = group_faqs(questions, llm, store, metadata_store)

    assert len(groups) == 1
    assert groups[0].count == 7
    assert len(groups[0].questions) <= 5


def test_group_faqs_redacts_names_from_returned_questions() -> None:
    llm = _RedactingLLM(embed_fn=_embed_fn)
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [FaqQuestionInput(id="q_name", text="Ask John Doe for VPN access")]

    groups = group_faqs(questions, llm, store, metadata_store)

    assert len(groups) == 1
    assert groups[0].question == "Ask [NAME] for VPN access"
    assert groups[0].questions == ["Ask [NAME] for VPN access"]

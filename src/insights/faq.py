"""Semantic grouping of recurring user questions into FAQ clusters.

Called by the backend's ``POST /insights/faq/refresh`` (pull-based, per
issue #66). ``/chat`` is stateless and this service does not retain question
history itself, so the backend sends the full set of questions to group on
every request.
"""

import logging
from dataclasses import dataclass, field

from ingestion.metadata_store import IngestionMetadataStore
from insights.redaction import redact_pii
from llm.base import LLMClient
from onboarding.similarity import cosine_similarity
from rag.retriever import retrieve
from store.base import VectorStore

logger = logging.getLogger(__name__)

# Cosine similarity above which two questions are folded into one FAQ group.
_SIMILARITY_THRESHOLD = 0.75
# Cap on redacted sample questions returned per group.
_MAX_SAMPLE_QUESTIONS = 5
# Retrieval settings used to find the documents that answer a group.
_DOCS_TOP_K = 5
_DOCS_MIN_SCORE = 0.3
# Cap on distinct documents returned per group.
_MAX_DOCUMENTS = 5


@dataclass(frozen=True)
class FaqQuestionInput:
    id: str
    text: str


@dataclass(frozen=True)
class FaqDocument:
    id: str
    title: str
    source: str | None


@dataclass(frozen=True)
class FaqGroup:
    question: str
    count: int
    questions: list[str]
    documents: list[FaqDocument]


@dataclass
class _Cluster:
    members: list[FaqQuestionInput] = field(default_factory=list[FaqQuestionInput])
    seed_embedding: list[float] = field(default_factory=list[float])


def _cluster_questions(
    questions: list[FaqQuestionInput], llm: LLMClient
) -> list[_Cluster]:
    """Greedily assign each question to the closest matching cluster.

    Each question is compared against every existing cluster's seed (first
    member's) embedding; it joins the best-matching cluster above the
    similarity threshold, or starts a new one otherwise. This leader-cluster
    approach mirrors the dedup pattern already used for onboarding step
    generation (``onboarding.generation.filter_semantic_duplicates``).
    """
    clusters: list[_Cluster] = []
    for question in questions:
        text = question.text.strip()
        if not text:
            continue
        embedding = llm.embed(text)

        best_cluster: _Cluster | None = None
        best_similarity = 0.0
        for cluster in clusters:
            similarity = cosine_similarity(embedding, cluster.seed_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None and best_similarity >= _SIMILARITY_THRESHOLD:
            best_cluster.members.append(question)
        else:
            clusters.append(_Cluster(members=[question], seed_embedding=embedding))

    return clusters


def _documents_for(
    representative_text: str,
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
) -> list[FaqDocument]:
    chunks = retrieve(
        question=representative_text,
        llm=llm,
        store=store,
        top_k=_DOCS_TOP_K,
        min_score=_DOCS_MIN_SCORE,
    )

    documents: dict[str, FaqDocument] = {}
    for chunk in chunks:
        if chunk.artifact_id in documents:
            continue
        record = metadata_store.get_artifact(chunk.artifact_id)
        documents[chunk.artifact_id] = FaqDocument(
            id=chunk.artifact_id,
            title=record.filename if record is not None else chunk.filename,
            source=record.source_type if record is not None else chunk.artifact_type,
        )
        if len(documents) >= _MAX_DOCUMENTS:
            break

    return list(documents.values())


def group_faqs(
    questions: list[FaqQuestionInput],
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
) -> list[FaqGroup]:
    clusters = _cluster_questions(questions, llm)

    # Redact every representative + sample question in a single batched LLM
    # call rather than one call per group.
    sample_texts: list[str] = []
    sample_bounds: list[tuple[int, int]] = []
    for cluster in clusters:
        seen: list[str] = []
        for member in cluster.members:
            if len(seen) >= _MAX_SAMPLE_QUESTIONS:
                break
            if member.text not in seen:
                seen.append(member.text)
        start = len(sample_texts)
        sample_texts.extend(seen)
        sample_bounds.append((start, start + len(seen)))

    redacted = redact_pii(sample_texts, llm)

    groups: list[FaqGroup] = []
    for cluster, (start, end) in zip(clusters, sample_bounds, strict=True):
        redacted_samples = redacted[start:end]
        representative_text = cluster.members[0].text
        representative_redacted = (
            redacted_samples[0] if redacted_samples else representative_text
        )
        groups.append(
            FaqGroup(
                question=representative_redacted,
                count=len(cluster.members),
                questions=redacted_samples,
                documents=_documents_for(
                    representative_text, llm, store, metadata_store
                ),
            )
        )

    groups.sort(key=lambda g: g.count, reverse=True)
    return groups

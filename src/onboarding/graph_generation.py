"""AI-authoring of competency graph proposals from the ingested corpus.

A batch, re-runnable job that proposes new ``Competency`` nodes (``SKILL``/
``CONCEPT``, per this slice's scope) and ``PREREQUISITE`` edges between them,
grounded in the ingested repo. It reuses the same retrieval layer
(:func:`rag.hybrid.hybrid_retrieve`) and idempotency mechanism
(:func:`onboarding.generation.corpus_fingerprint`) as blueprint generation.

Deliberately simpler than blueprint generation: the backend's competency graph
today can only grow (see :mod:`onboarding.graph_models`), so there is no
invariant-protection gate here -- a proposal run only ever adds new
candidates, it never redrafts or replaces what already exists. The real
constraint is dedup: a competency already in the graph must not be re-proposed
(exact key match), and a near-duplicate must not slip through either
(embedding similarity, same threshold blueprint generation uses).

The backend has no persisted proposal-lifecycle for competencies/edges yet
(unlike ``Blueprint``'s DRAFT/PROPOSED/ACTIVE/ARCHIVED) -- that is a real gap
this job's caller must eventually address (see backend issue #7's "Done when"
list, deferred). Until then, idempotency is driven by an explicit
``last_fingerprint`` the caller passes in and is responsible for persisting,
since there is no "active proposal" object here to carry it the way
``Blueprint.provenance.corpus_fingerprint`` does.
"""

import json
import logging
import re
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationError

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.generation import corpus_fingerprint
from onboarding.graph_models import (
    ActiveCompetency,
    ActiveEdge,
    GraphProposalOutcome,
    GraphProvenance,
    ProposedCompetency,
    ProposedEdge,
)
from onboarding.models import CitationRef
from onboarding.similarity import SIMILARITY_THRESHOLD, cosine_similarity, step_text
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K = 20
_MIN_SCORE = 0.3
_QUERY = (
    "core skills, concepts, technologies, architecture patterns and domain "
    "knowledge required to work productively in this codebase"
)
_KEY_RE = re.compile(r"[^a-z0-9]+")


class GenerationError(Exception):
    """Raised when the LLM output for a graph proposal cannot be parsed/validated."""


# --- LLM payload -------------------------------------------------------------


class _GenCompetency(BaseModel):
    key: str
    label: str
    description: str = ""
    kind: str = "SKILL"
    repo_ref: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)


class _GenEdge(BaseModel):
    from_key: str
    to_key: str
    rationale: str = ""


class _GenPayload(BaseModel):
    competencies: list[_GenCompetency] = Field(default_factory=list[_GenCompetency])
    edges: list[_GenEdge] = Field(default_factory=list[_GenEdge])


# --- prompt / parsing ---------------------------------------------------------


def _normalize_key(raw: str) -> str:
    """Kebab-case a proposed key so it matches the backend's stable-key convention."""
    return _KEY_RE.sub("-", raw.strip().lower()).strip("-")


def _build_prompt(
    chunks: list[ScoredChunk], active: list[ActiveCompetency]
) -> list[Message]:
    evidence = "\n".join(f"[{c.id}] ({c.filename}) {c.text}" for c in chunks)
    exclusion = ""
    if active:
        existing = "\n".join(f"- {c.key}: {c.label}" for c in active)
        exclusion = (
            "\nThese competencies already exist in the graph -- do NOT propose "
            "them again under a new key, even if phrased differently. You MAY "
            "reference their key as a prerequisite edge endpoint.\n\n"
            f"Existing competencies:\n{existing}\n"
        )
    system = (
        "You propose nodes and edges for a team's competency graph from its "
        "knowledge base. You are given evidence snippets, each prefixed with "
        "its chunk id in square brackets.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences). Rules:\n"
        "1. Propose competency nodes of kind SKILL (a tool/language/technology) "
        "or CONCEPT (a domain/architecture idea specific to this codebase).\n"
        "2. Every competency MUST cite at least one chunk id from the evidence; "
        "do not invent sources.\n"
        "3. Each competency needs a short, stable, kebab-case `key` (e.g. "
        "'kotlin', 'our-domain-model') distinct from any existing key.\n"
        "4. `repo_ref` is an optional pointer to the file/path the competency "
        "is grounded in, when the evidence makes one obvious.\n"
        "5. Propose PREREQUISITE edges only where a genuine dependency exists "
        "-- the `to_key` competency requires the `from_key` one first. Give "
        "each edge a one-sentence `rationale`.\n"
        f"{exclusion}"
        'JSON schema: {"competencies": [{"key": str, "label": str, '
        '"description": str, "kind": "SKILL"|"CONCEPT", "repo_ref": str|null, '
        '"chunk_ids": [str]}], "edges": [{"from_key": str, "to_key": str, '
        '"rationale": str}]}'
    )
    user = f"Evidence:\n{evidence}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _parse_payload(raw: str) -> _GenPayload:
    try:
        return _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid generation output: {exc}") from exc


# --- dedup ---------------------------------------------------------------------


def _filter_duplicate_competencies(
    candidates: list[tuple[_GenCompetency, list[CitationRef]]],
    active: list[ActiveCompetency],
    llm: LLMClient,
) -> list[tuple[_GenCompetency, list[CitationRef]]]:
    """Drop exact-key and near-duplicate (embedding similarity) proposals.

    Mirrors ``generation.filter_semantic_duplicates``, adapted to competencies:
    a proposal is dropped if its key exactly matches an active competency, or
    if its embedding is too close to an active competency's or an
    already-kept proposal's (first occurrence wins).
    """
    active_keys = {c.key for c in active}
    seen_embeddings: list[list[float]] = [
        llm.embed(step_text(c.label, c.description)) for c in active
    ]

    kept: list[tuple[_GenCompetency, list[CitationRef]]] = []
    seen_keys: set[str] = set()
    for competency, citations in candidates:
        key = _normalize_key(competency.key)
        if not key or key in active_keys or key in seen_keys:
            continue
        embedding = llm.embed(step_text(competency.label, competency.description))
        max_sim = max(
            (cosine_similarity(embedding, prior) for prior in seen_embeddings),
            default=0.0,
        )
        if max_sim >= SIMILARITY_THRESHOLD:
            logger.info(
                "Dropped duplicate competency proposal %r (sim=%.2f)",
                competency.label,
                max_sim,
            )
            continue
        seen_keys.add(key)
        seen_embeddings.append(embedding)
        kept.append((competency, citations))
    return kept


# --- job -----------------------------------------------------------------------


def generate_competency_graph(
    llm: LLMClient,
    store: VectorStore,
    *,
    active_competencies: list[ActiveCompetency] | None = None,
    active_edges: list[ActiveEdge] | None = None,
    last_fingerprint: str | None = None,
) -> GraphProposalOutcome:
    """Propose new competency nodes/edges for the backend to persist and a PM to review.

    ``active_competencies``/``active_edges`` are the backend's current live graph --
    they drive dedup (never re-propose an existing key) and are valid edge
    endpoints. ``last_fingerprint`` is whatever fingerprint the caller recorded
    from the previous run (idempotency); this service holds no state of its own.
    """
    active = active_competencies or []
    active_edge_list = active_edges or []
    fingerprint = corpus_fingerprint(store)

    if last_fingerprint is not None and last_fingerprint == fingerprint:
        return GraphProposalOutcome(
            status="unchanged", notes=["corpus unchanged since last proposal run"]
        )

    if store.count() == 0:
        return GraphProposalOutcome(status="skipped", notes=["corpus is empty"])

    chunks = hybrid_retrieve(
        question=_QUERY,
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=BM25IndexCache(),
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
    )
    if not chunks:
        return GraphProposalOutcome(
            status="skipped", notes=["no grounding evidence retrieved"]
        )

    raw = llm.generate(_build_prompt(chunks, active))
    try:
        payload = _parse_payload(raw)
    except GenerationError as exc:
        logger.warning("Graph proposal generation failed: %s", exc)
        return GraphProposalOutcome(status="skipped", notes=[str(exc)])

    chunks_by_id = {c.id: c for c in chunks}
    grounded: list[tuple[_GenCompetency, list[CitationRef]]] = []
    for item in payload.competencies:
        citations = resolve_citations(item.chunk_ids, chunks_by_id)
        if not citations:
            continue  # grounding gate: drop ungrounded proposals
        if item.kind not in ("SKILL", "CONCEPT"):
            continue
        grounded.append((item, citations))

    kept = _filter_duplicate_competencies(grounded, active, llm)
    proposed_competencies = [
        ProposedCompetency(
            key=_normalize_key(item.key),
            label=item.label,
            description=item.description,
            kind=item.kind,  # type: ignore[arg-type]
            repo_ref=item.repo_ref,
            citations=citations,
        )
        for item, citations in kept
    ]

    known_keys = {c.key for c in active} | {c.key for c in proposed_competencies}
    existing_edges = {(e.from_key, e.to_key) for e in active_edge_list}
    proposed_edges: list[ProposedEdge] = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in payload.edges:
        from_key = _normalize_key(edge.from_key)
        to_key = _normalize_key(edge.to_key)
        if from_key == to_key or from_key not in known_keys or to_key not in known_keys:
            continue
        pair = (from_key, to_key)
        if pair in existing_edges or pair in seen_edges:
            continue
        seen_edges.add(pair)
        proposed_edges.append(
            ProposedEdge(from_key=from_key, to_key=to_key, rationale=edge.rationale)
        )

    if not proposed_competencies:
        return GraphProposalOutcome(
            status="skipped",
            chunks_retrieved=len(chunks),
            notes=["no grounded, non-duplicate competencies proposed"],
        )

    return GraphProposalOutcome(
        status="proposed",
        competencies=proposed_competencies,
        edges=proposed_edges,
        provenance=GraphProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
        ),
        chunks_retrieved=len(chunks),
    )

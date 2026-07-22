"""AI-authoring of competency graph proposals from the ingested corpus.

A batch, re-runnable job that proposes new ``Competency`` nodes (``SKILL``/
``CONCEPT``, per this slice's scope) and the edges between them, grounded in
the ingested repo. It reuses the same retrieval layer
(:func:`rag.hybrid.hybrid_retrieve`) and idempotency mechanism
(:func:`onboarding.generation.corpus_fingerprint`) as blueprint generation.

Nodes and edges are two separate LLM calls. Asking for both at once reliably
produced a scatter -- live testing yielded 10 competencies and 3 edges, with
seven nodes related to nothing -- because the model spends a single call's
budget on nodes and treats edges as an afterthought. The second pass sees the
finished node list and does nothing else, so relationships compete with each
other for attention rather than with node generation. Consequently the corpus
fingerprint short-circuits the *node* pass only: relationships between nodes
already in the graph are not a function of the corpus, and a graph proposed
before this pass existed is precisely the one that needs re-running.

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
from collections.abc import Generator
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
from onboarding.progress import ProgressEvent, ProgressStream, drain
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


class _GenPayload(BaseModel):
    competencies: list[_GenCompetency] = Field(default_factory=list[_GenCompetency])


class _GenEdge(BaseModel):
    from_key: str
    to_key: str
    kind: str = "PREREQUISITE"
    rationale: str = ""


class _GenIsolated(BaseModel):
    key: str
    reason: str = ""


class _GenEdgePayload(BaseModel):
    """Output of the dedicated edge pass.

    ``tiers`` is a reasoning scaffold, not persisted: asking for a
    foundational-to-advanced layering before edges is what makes the model
    think in a progression rather than in isolated topics. ``isolated`` forces
    the model to *justify* leaving a node unconnected instead of defaulting to
    it -- the failure mode that produced a 10-node/3-edge scatter.
    """

    tiers: list[list[str]] = Field(default_factory=list[list[str]])
    edges: list[_GenEdge] = Field(default_factory=list[_GenEdge])
    isolated: list[_GenIsolated] = Field(default_factory=list[_GenIsolated])


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
        "You propose nodes for a team's competency graph from its "
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
        "5. Relationships between competencies are decided separately -- "
        "propose nodes only.\n"
        f"{exclusion}"
        'JSON schema: {"competencies": [{"key": str, "label": str, '
        '"description": str, "kind": "SKILL"|"CONCEPT", "repo_ref": str|null, '
        '"chunk_ids": [str]}]}'
    )
    user = f"Evidence:\n{evidence}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _build_edge_prompt(nodes: list[tuple[str, str, str, bool]]) -> list[Message]:
    """Prompt the relationships pass, given every node the graph will contain.

    Runs after nodes are settled so relationships compete with each other for
    the model's attention instead of competing with node generation -- the
    single-call version reliably spent its budget on nodes and proposed almost
    no edges.
    """
    listing = "\n".join(
        f"- {key}: {label}"
        + (f" -- {description}" if description else "")
        + ("  [new]" if is_new else "")
        for key, label, description, is_new in nodes
    )
    system = (
        "You lay out the structure of a team's competency graph: which "
        "competencies build on which. The nodes are already decided -- your "
        "only job is the relationships between them.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences).\n\n"
        "Work in two steps:\n"
        "1. `tiers`: order the competencies into layers from foundational to "
        "advanced, each layer a list of keys. Every key appears exactly once.\n"
        "2. `edges`: connect them, mostly within a tier or between adjacent "
        "tiers.\n\n"
        "Two edge kinds, and the difference matters:\n"
        "- PREREQUISITE (`from_key` must be held before `to_key` can be "
        "started) GATES the graph. Use it only for a genuine hard dependency. "
        "Long prerequisite chains delay a new hire's first contribution, so "
        "keep them short and shallow.\n"
        "- RELATED is structure without gating: builds on, part of, commonly "
        "used together, same area of the system. Use it freely -- this is "
        "where most relationships belong.\n\n"
        "What a healthy graph looks like: a handful of foundational roots, "
        "and nearly every other node connected to at least one other node. A "
        "node related to nothing is suspicious, not the default. If you truly "
        "cannot connect a node, list it in `isolated` with a one-sentence "
        "reason -- do not silently leave it out.\n\n"
        "Edges must be acyclic: never propose a path that leads back to where "
        "it started. Give every edge a one-sentence `rationale`.\n\n"
        'JSON schema: {"tiers": [[str]], "edges": [{"from_key": str, '
        '"to_key": str, "kind": "PREREQUISITE"|"RELATED", "rationale": str}], '
        '"isolated": [{"key": str, "reason": str}]}'
    )
    user = f"Competencies:\n{listing}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _parse_payload(raw: str) -> _GenPayload:
    try:
        return _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid generation output: {exc}") from exc


def _parse_edge_payload(raw: str) -> _GenEdgePayload:
    try:
        return _GenEdgePayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid edge output: {exc}") from exc


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


# --- edges ---------------------------------------------------------------------


def _reaches(start: str, target: str, adjacency: dict[str, set[str]]) -> bool:
    """Whether ``target`` is reachable from ``start`` along prerequisite edges."""
    seen: set[str] = set()
    stack: list[str] = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return False


def _propose_edges(
    llm: LLMClient,
    *,
    active: list[ActiveCompetency],
    proposed: list[ProposedCompetency],
    active_edges: list[ActiveEdge],
) -> tuple[list[ProposedEdge], list[str]]:
    """Run the dedicated relationships pass over every node the graph will contain.

    Returns the accepted edges plus notes for the PM (isolation reasons the
    model gave, and anything this function dropped). Never raises: a failed
    edge pass degrades to "no new edges", it does not lose the node proposals.
    """
    nodes: list[tuple[str, str, str, bool]] = [
        (c.key, c.label, c.description, False) for c in active
    ] + [(c.key, c.label, c.description, True) for c in proposed]
    if len(nodes) < 2:
        return [], []

    try:
        payload = _parse_edge_payload(llm.generate(_build_edge_prompt(nodes)))
    except GenerationError as exc:
        logger.warning("Competency edge pass failed: %s", exc)
        return [], [f"edge proposal pass failed: {exc}"]

    known_keys = {key for key, _, _, _ in nodes}
    existing_pairs = {(e.from_key, e.to_key) for e in active_edges}
    # Cycles are only meaningful for the gating kind; RELATED edges express
    # association, so a loop of them locks nothing.
    prereq_adjacency: dict[str, set[str]] = {}
    for edge in active_edges:
        if edge.kind == "PREREQUISITE":
            prereq_adjacency.setdefault(edge.from_key, set()).add(edge.to_key)

    accepted: list[ProposedEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    cycles_dropped = 0
    for edge in payload.edges:
        from_key = _normalize_key(edge.from_key)
        to_key = _normalize_key(edge.to_key)
        kind = edge.kind if edge.kind in ("PREREQUISITE", "RELATED") else "RELATED"
        if from_key == to_key or from_key not in known_keys or to_key not in known_keys:
            continue
        pair = (from_key, to_key)
        if pair in existing_pairs or pair in seen_pairs:
            continue
        if kind == "RELATED" and (
            (to_key, from_key) in seen_pairs or (to_key, from_key) in existing_pairs
        ):
            continue  # association is symmetric; one direction is enough
        if kind == "PREREQUISITE" and _reaches(to_key, from_key, prereq_adjacency):
            cycles_dropped += 1
            continue
        if kind == "PREREQUISITE":
            prereq_adjacency.setdefault(from_key, set()).add(to_key)
        seen_pairs.add(pair)
        accepted.append(
            ProposedEdge(
                from_key=from_key,
                to_key=to_key,
                kind=kind,  # type: ignore[arg-type]
                rationale=edge.rationale,
            )
        )

    notes: list[str] = []
    if cycles_dropped:
        notes.append(
            f"dropped {cycles_dropped} prerequisite edge(s) that formed a cycle"
        )
    for isolated in payload.isolated:
        key = _normalize_key(isolated.key)
        if key in known_keys:
            reason = isolated.reason or "no reason given"
            notes.append(f"unconnected: {key} -- {reason}")

    connected = {key for edge in accepted for key in (edge.from_key, edge.to_key)}
    connected |= {key for pair in existing_pairs for key in pair}
    orphans = sorted(known_keys - connected)
    if orphans:
        notes.append(
            f"{len(orphans)} competency/-ies remain unconnected: " + ", ".join(orphans)
        )

    return accepted, notes


# --- job -----------------------------------------------------------------------


def stream_competency_graph(
    llm: LLMClient,
    store: VectorStore,
    *,
    active_competencies: list[ActiveCompetency] | None = None,
    active_edges: list[ActiveEdge] | None = None,
    last_fingerprint: str | None = None,
) -> Generator[ProgressEvent, None, GraphProposalOutcome]:
    """Propose the graph, yielding live progress and returning the final outcome.

    This is the single implementation: :func:`generate_competency_graph` drives it
    to completion for the non-streaming path, and the streaming route relays its
    events. So the proposal a PM watches assemble is byte-for-byte the proposal the
    batch call would have produced — the stream is a view, never a second answer.

    The graph literally builds: the node pass emits each grounded, deduped
    competency as an ``item`` the instant it clears its gate, then the dedicated
    edge pass emits each accepted relationship. An ``item`` is a promise of
    validation — if it was streamed, it is in the persisted proposal.
    """
    progress = ProgressStream("competency_graph")
    active = active_competencies or []
    active_edge_list = active_edges or []
    fingerprint = corpus_fingerprint(store)
    notes: list[str] = []

    if store.count() == 0:
        outcome = GraphProposalOutcome(status="skipped", notes=["corpus is empty"])
        yield progress.warning("The project has no indexed material yet")
        yield progress.done("No competencies could be proposed", _dump(outcome))
        return outcome

    # An unchanged corpus can hold no *new* nodes, but the relationships between
    # the nodes already in the graph are not a function of the corpus -- a graph
    # generated before the dedicated edge pass existed is exactly the case that
    # needs re-running. So the fingerprint short-circuits the node pass only.
    corpus_unchanged = last_fingerprint is not None and last_fingerprint == fingerprint
    proposed_competencies: list[ProposedCompetency] = []
    chunks: list[ScoredChunk] = []

    if corpus_unchanged:
        notes.append("corpus unchanged since last proposal run; nodes not re-proposed")
        yield progress.stage(
            "retrieving", "Corpus unchanged — proposing relationships only"
        )
    else:
        yield progress.stage("retrieving", "Searching the corpus for competencies")
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
            outcome = GraphProposalOutcome(
                status="skipped", notes=["no grounding evidence retrieved"]
            )
            yield progress.warning("Nothing in the corpus grounded a competency")
            yield progress.done("No competencies could be proposed", _dump(outcome))
            return outcome
        yield progress.stage(
            "grounding", f"Proposing competencies from {len(chunks)} source(s)"
        )
        try:
            proposed_competencies = _propose_competencies(llm, chunks, active)
        except GenerationError as exc:
            logger.warning("Graph proposal generation failed: %s", exc)
            outcome = GraphProposalOutcome(status="skipped", notes=[str(exc)])
            yield progress.warning("The proposed competencies could not be read")
            yield progress.done("No competencies could be proposed", _dump(outcome))
            return outcome
        for competency in proposed_competencies:
            yield progress.item(
                competency.model_dump(mode="json"),
                f"Competency: {competency.label}",
            )

    yield progress.stage("linking", "Working out how the competencies relate")
    proposed_edges, edge_notes = _propose_edges(
        llm,
        active=active,
        proposed=proposed_competencies,
        active_edges=active_edge_list,
    )
    for edge in proposed_edges:
        yield progress.item(
            edge.model_dump(mode="json"),
            f"{edge.from_key} → {edge.to_key} ({edge.kind})",
        )
    notes.extend(edge_notes)

    if not proposed_competencies and not proposed_edges:
        outcome = GraphProposalOutcome(
            status="unchanged" if corpus_unchanged else "skipped",
            chunks_retrieved=len(chunks),
            notes=notes or ["no grounded, non-duplicate competencies proposed"],
        )
        yield progress.done("Nothing new to propose", _dump(outcome))
        return outcome

    outcome = GraphProposalOutcome(
        status="proposed",
        competencies=proposed_competencies,
        edges=proposed_edges,
        provenance=GraphProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
        ),
        chunks_retrieved=len(chunks),
        notes=notes,
    )
    yield progress.done(
        f"Proposed {len(proposed_competencies)} competency/-ies "
        f"and {len(proposed_edges)} relationship(s)",
        _dump(outcome),
    )
    return outcome


def _dump(outcome: GraphProposalOutcome) -> dict[str, object]:
    """The outcome as a JSON-safe dict for a ``done`` event's ``result``."""
    return outcome.model_dump(mode="json")


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

    Backed by :func:`stream_competency_graph` so the non-streaming and streaming
    paths are the same computation.
    """
    return drain(
        stream_competency_graph(
            llm,
            store,
            active_competencies=active_competencies,
            active_edges=active_edges,
            last_fingerprint=last_fingerprint,
        )
    )


def _propose_competencies(
    llm: LLMClient, chunks: list[ScoredChunk], active: list[ActiveCompetency]
) -> list[ProposedCompetency]:
    """Run the node pass: grounded, deduped competency proposals."""
    payload = _parse_payload(llm.generate(_build_prompt(chunks, active)))

    chunks_by_id = {c.id: c for c in chunks}
    grounded: list[tuple[_GenCompetency, list[CitationRef]]] = []
    for item in payload.competencies:
        citations = resolve_citations(item.chunk_ids, chunks_by_id)
        if not citations:
            continue  # grounding gate: drop ungrounded proposals
        if item.kind not in ("SKILL", "CONCEPT"):
            continue
        grounded.append((item, citations))

    return [
        ProposedCompetency(
            key=_normalize_key(item.key),
            label=item.label,
            description=item.description,
            kind=item.kind,  # type: ignore[arg-type]
            repo_ref=item.repo_ref,
            citations=citations,
        )
        for item, citations in _filter_duplicate_competencies(grounded, active, llm)
    ]

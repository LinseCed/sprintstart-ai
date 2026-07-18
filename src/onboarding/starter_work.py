"""AI-authoring of starter-work pool candidates from open GitHub issues.

Phase 4 (ai issue 5): source the contribution (goal) nodes a hire's path
terminates in. A batch, re-runnable job -- offline/authoring-time, not on the
hire's request path -- that mines the ingested corpus's open GitHub issues for
safely-scoped starter tasks and proposes them as a candidate pool for PM
curation. Proposal-only, never auto-published: mirrors
:mod:`onboarding.graph_generation`'s relationship to the backend, including
reuse of its idempotency mechanism (:func:`onboarding.generation.corpus_fingerprint`).

Candidate sourcing is deterministic, not LLM-driven: only ``ISSUE`` artifacts
with ``state == "OPEN"`` are ever considered (see
:class:`ingestion.metadata_store.ArtifactRecord`'s docstring) -- closed issues
are excluded before the LLM ever sees them, rather than relying on it to
notice. The LLM's role is judgment, not filtering: given each open issue's own
text, it assesses whether the issue is *safely scoped* for a new hire (small,
clear acceptance criteria, no cross-cutting blast radius) and tags the
competencies it exercises. There is no free-form retrieval step here the way
there is in :mod:`onboarding.graph_generation`, since the source of truth for
each candidate is the issue itself, not a synthesized answer -- grounding is
"this proposal came from this issue's own chunks", not citation resolution
against a broader corpus.

"TODOs in code" and "small-surface modules" (the other candidate sources
issue #5 asks for) are out of scope for this slice: unlike an issue, a TODO
comment or a small module has no ingested owner or acceptance criteria to
ground a proposal in, so mining them needs its own grounding strategy.
Deferred rather than guessed at.
"""

import json
import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationError

from ingestion.metadata_store import IngestionMetadataStore
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.generation import corpus_fingerprint
from onboarding.graph_models import ProposalStatus
from onboarding.models import CitationRef
from rag.types import Chunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_MAX_CHUNKS_PER_ISSUE = 20


class GenerationError(Exception):
    """Raised when the LLM output for a starter-work proposal can't be parsed."""


# --- domain models -------------------------------------------------------------


class StarterTaskCandidate(BaseModel):
    """One open issue considered for the starter-work pool."""

    source_id: str
    title: str
    text: str
    source_url: str | None = None
    labels: list[str] = Field(default_factory=list[str])


class ProposedStarterTask(BaseModel):
    """A candidate starter task grounded in one open issue, for PM curation."""

    source_id: str
    title: str
    summary: str = ""
    competency_keys: list[str] = Field(default_factory=list[str])
    rationale: str = ""
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class StarterWorkProvenance(BaseModel):
    """Why a mining run looks the way it does; mirrors ``GraphProvenance``."""

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class StarterWorkOutcome(BaseModel):
    """Result of one starter-work mining run."""

    status: ProposalStatus
    tasks: list[ProposedStarterTask] = Field(default_factory=list[ProposedStarterTask])
    provenance: StarterWorkProvenance | None = None
    candidates_considered: int = 0
    notes: list[str] = Field(default_factory=list[str])


# --- LLM payload -----------------------------------------------------------------


class _GenTask(BaseModel):
    source_id: str = ""
    safely_scoped: bool = False
    summary: str = ""
    competency_keys: list[str] = Field(default_factory=list[str])
    rationale: str = ""


class _GenPayload(BaseModel):
    tasks: list[_GenTask] = Field(default_factory=list[_GenTask])


# --- prompt / parsing --------------------------------------------------------------


def _derive_title(text: str, fallback: str) -> str:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip() or fallback
    return fallback


def _issue_block(candidate: StarterTaskCandidate) -> str:
    label_note = f" [labels: {', '.join(candidate.labels)}]" if candidate.labels else ""
    return f"[{candidate.source_id}]{label_note}\n{candidate.text}"


def _build_prompt(
    candidates: list[StarterTaskCandidate], known_competency_keys: list[str]
) -> list[Message]:
    issues = "\n\n---\n\n".join(_issue_block(c) for c in candidates)
    known = ""
    if known_competency_keys:
        known = (
            "\nKnown competency keys you may tag a task with (do not invent new "
            f"ones): {', '.join(sorted(known_competency_keys))}\n"
        )
    system = (
        "You review a software team's open GitHub issues to build a curated "
        "pool of starter tasks for new hires. You are given each issue's full "
        "text, prefixed with its source id in square brackets.\n\n"
        "For each issue, decide whether it is safely scoped for a first-time "
        "contributor:\n"
        "- 'safely_scoped' is true only if the issue has a small, well-defined "
        "surface (touches one module/area, not a cross-cutting change), a "
        "clear description of what 'done' looks like, and no dependency on "
        "context a new hire wouldn't have.\n"
        "- Reject (safely_scoped=false) anything vague, large, architectural, "
        "or that reads as blocked/needs-discussion.\n"
        "- 'summary' is a one or two sentence restatement of the task for a "
        "hire browsing the pool -- do not just repeat the issue title.\n"
        "- 'competency_keys' tags the skills/concepts this task would "
        "exercise, chosen only from the known keys below when a list is "
        "given.\n"
        "- 'rationale' is one sentence on why the task is (or isn't) safely "
        "scoped.\n"
        f"{known}\n"
        "Return STRICT JSON only (no prose, no markdown fences), one entry "
        "per issue you were given, correlated by 'source_id':\n"
        '{"tasks": [{"source_id": str, "safely_scoped": bool, "summary": str, '
        '"competency_keys": [str], "rationale": str}]}'
    )
    user = f"Open issues:\n\n{issues}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _parse_payload(raw: str) -> _GenPayload:
    try:
        return _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid starter-work output: {exc}") from exc


# --- candidate sourcing ------------------------------------------------------------


def _issue_text(chunks: list[Chunk]) -> str:
    ordered = sorted(chunks, key=lambda c: c.position or 0)
    return "\n".join(c.text for c in ordered)


def _load_candidates(
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    *,
    exclude_source_ids: set[str],
) -> list[StarterTaskCandidate]:
    candidates: list[StarterTaskCandidate] = []
    for artifact in metadata_store.list_artifacts(status="completed"):
        if artifact.artifact_type != "ISSUE":
            continue
        if (artifact.state or "").upper() != "OPEN":
            continue
        if artifact.source_id is None or artifact.source_id in exclude_source_ids:
            continue

        chunks = store.list_chunks_by_artifact(artifact.id, limit=_MAX_CHUNKS_PER_ISSUE)
        if not chunks:
            continue

        text = _issue_text(chunks)
        if not text.strip():
            continue

        candidates.append(
            StarterTaskCandidate(
                source_id=artifact.source_id,
                title=_derive_title(text, artifact.filename),
                text=text,
                source_url=artifact.source_url,
                labels=artifact.labels,
            )
        )
    return candidates


# --- job -----------------------------------------------------------------------


def generate_starter_work_pool(
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    *,
    active_source_ids: list[str] | None = None,
    active_competency_keys: list[str] | None = None,
    last_fingerprint: str | None = None,
) -> StarterWorkOutcome:
    """Propose safely-scoped starter tasks from open GitHub issues.

    ``active_source_ids`` are issues already in the backend's starter-work
    pool (proposed or approved) -- drives dedup, an issue already pooled is
    never re-proposed. ``active_competency_keys`` are the backend's live
    competency graph keys; a proposed task's competency tags are grounded
    against this set (dropped, not invented, when the tag falls outside it) --
    when no keys are supplied there is nothing to validate against, so tags
    are kept as the LLM proposed them. ``last_fingerprint`` mirrors
    :func:`onboarding.graph_generation.generate_competency_graph`'s
    corpus-wide idempotency.
    """
    fingerprint = corpus_fingerprint(store)
    if last_fingerprint is not None and last_fingerprint == fingerprint:
        return StarterWorkOutcome(
            status="unchanged", notes=["corpus unchanged since last mining run"]
        )

    if store.count() == 0:
        return StarterWorkOutcome(status="skipped", notes=["corpus is empty"])

    candidates = _load_candidates(
        store, metadata_store, exclude_source_ids=set(active_source_ids or [])
    )
    if not candidates:
        return StarterWorkOutcome(
            status="skipped", notes=["no open, unpooled issues found"]
        )

    raw = llm.generate(_build_prompt(candidates, active_competency_keys or []))
    try:
        payload = _parse_payload(raw)
    except GenerationError as exc:
        logger.warning("Starter-work mining generation failed: %s", exc)
        return StarterWorkOutcome(
            status="skipped",
            candidates_considered=len(candidates),
            notes=[str(exc)],
        )

    candidates_by_id = {c.source_id: c for c in candidates}
    known_keys = set(active_competency_keys) if active_competency_keys else None

    tasks: list[ProposedStarterTask] = []
    seen: set[str] = set()
    for item in payload.tasks:
        if not item.safely_scoped:
            continue
        candidate = candidates_by_id.get(item.source_id)
        if candidate is None or candidate.source_id in seen:
            continue
        seen.add(candidate.source_id)

        keys = [
            k for k in item.competency_keys if known_keys is None or k in known_keys
        ]
        citation = CitationRef(
            filename=candidate.title,
            chunk_id=candidate.source_id,
            source_url=candidate.source_url,
        )
        tasks.append(
            ProposedStarterTask(
                source_id=candidate.source_id,
                title=candidate.title,
                summary=item.summary or candidate.title,
                competency_keys=keys,
                rationale=item.rationale,
                citations=[citation],
            )
        )

    if not tasks:
        return StarterWorkOutcome(
            status="skipped",
            candidates_considered=len(candidates),
            notes=["no candidate judged safely scoped"],
        )

    return StarterWorkOutcome(
        status="proposed",
        tasks=tasks,
        provenance=StarterWorkProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
        ),
        candidates_considered=len(candidates),
    )

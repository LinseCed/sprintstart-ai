"""AI-authoring of onboarding blueprints from the ingested corpus (issue #110).

A batch, re-runnable job that drafts/updates blueprints (``scope: global`` and
``scope: area:<name>``) as ``source: generated`` artifacts. It reuses the
existing retrieval layer (:func:`rag.hybrid.hybrid_retrieve`) and the
``LLMClient`` abstraction — there is no separate ingest or retrieval path.

The job is deliberately conservative:

* **Grounded** — every proposed step must cite at least one retrieved chunk;
  ungrounded steps are dropped (consistent with the path-generation grounding
  gate in ``onboarding/pipeline.py``).
* **Idempotent** — a blueprint records the ``corpus_fingerprint`` it was drafted
  from. Re-running against an unchanged corpus is a no-op (no draft churn).
* **Invariant-safe** — it may not remove or downgrade a human-owned step
  (``required`` or ``invariant`` in the active blueprint). Such changes are
  blocked: the protected step is re-injected and the draft is escalated, never
  silently applied.

It writes to the review queue (``onboarding/drafts.py``); promotion to active is
a separate, human-approved step.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding import drafts
from onboarding.blueprints import load_blueprints
from onboarding.models import (
    Blueprint,
    BlueprintProvenance,
    BlueprintStep,
    CitationRef,
    Skeleton,
    SkeletonRef,
    StepRecord,
)
from onboarding.registry import load_pool, save_pool, upsert_step
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

OutcomeStatus = Literal["created", "updated", "unchanged", "escalated", "skipped"]

_TOP_K = 12
_MIN_SCORE = 0.3


class GenerationError(Exception):
    """Raised when the LLM output for a scope cannot be parsed/validated."""


class GenerationOutcome(BaseModel):
    scope: str
    status: OutcomeStatus
    draft_version: str | None = None
    chunks_retrieved: int = 0
    steps_drafted: int = 0
    model: str | None = None
    notes: list[str] = Field(default_factory=list)


# --- LLM payload -----------------------------------------------------------


class _GenStep(BaseModel):
    title: str
    description: str = ""
    requirement: Literal["required", "recommended"] = "recommended"
    tags: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class _GenPayload(BaseModel):
    steps: list[_GenStep] = []


# --- corpus fingerprint (idempotency) --------------------------------------


def corpus_fingerprint(store: VectorStore) -> str:
    """Stable hash of the corpus contents; changes iff the corpus changes."""
    digest = hashlib.sha256()
    for chunk in sorted(store.all_chunks(), key=lambda c: c.id):
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


# --- scope helpers ---------------------------------------------------------


def default_scopes() -> list[str]:
    """``global`` plus every area scope present among existing blueprints."""
    scopes = {b.scope for b in load_blueprints()}
    scopes.add("global")
    return sorted(scopes)


def _scope_label(scope: str) -> str:
    if scope == "global":
        return "everyone on the team, regardless of working area"
    if scope.startswith("area:"):
        return f"new team members working in {scope.split(':', 1)[1]}"
    return scope


def _scope_query(scope: str) -> str:
    if scope == "global":
        return "team onboarding essentials getting started setup access"
    if scope.startswith("area:"):
        return f"{scope.split(':', 1)[1]} onboarding essentials setup workflow"
    return f"{scope} onboarding"


# --- prompt / parsing ------------------------------------------------------


def _build_prompt(
    scope: str,
    chunks: list[ScoredChunk],
    global_steps: list[BlueprintStep] | None = None,
) -> list[Message]:
    evidence = "\n".join(f"[{c.id}] ({c.filename}) {c.text}" for c in chunks)
    exclusion = ""
    if global_steps:
        covered = "\n".join(f"- {s.title}" for s in global_steps)
        exclusion = (
            "\n3. The global blueprint already covers these steps. Do NOT "
            "duplicate or rephrase them; only propose steps specific to this "
            f"area:\n{covered}\n"
        )
    system = (
        "You draft an onboarding blueprint for a software team from its knowledge "
        "base. You are given evidence snippets, each prefixed with its chunk id in "
        "square brackets.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences). Propose a concise, "
        f"ordered list of onboarding steps for {_scope_label(scope)}. Rules:\n"
        "1. Every step MUST reference at least one chunk id from the evidence; do "
        "not invent sources.\n"
        "2. Mark foundational/mandatory steps as 'required', others as "
        "'recommended'.\n"
        f"{exclusion}\n"
        'JSON schema: {"steps": [{"title": str, "description": str, '
        '"requirement": "required"|"recommended", "tags": [str], '
        '"chunk_ids": [str]}]}'
    )
    user = f"Scope: {scope}\n\nEvidence:\n{evidence}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _draft_steps(
    scope: str,
    chunks: list[ScoredChunk],
    llm: LLMClient,
    pool: dict[str, StepRecord],
    global_steps: list[BlueprintStep] | None = None,
) -> list[SkeletonRef]:
    """Ask the LLM for steps, keep the grounded ones, and upsert them.

    Grounded steps are written into ``pool`` (deduplicating by content
    fingerprint) and returned as ordered skeleton references carrying the
    proposed ``requirement``.
    """
    raw = llm.generate(_build_prompt(scope, chunks, global_steps))
    try:
        payload = _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid generation output: {exc}") from exc

    chunks_by_id = {c.id: c for c in chunks}

    def resolve(chunk_ids: list[str]) -> list[CitationRef]:
        refs: list[CitationRef] = []
        seen: set[str] = set()
        for cid in chunk_ids:
            chunk = chunks_by_id.get(cid)
            if chunk is not None and chunk.id not in seen:
                refs.append(CitationRef(filename=chunk.filename, chunk_id=chunk.id))
                seen.add(chunk.id)
        return refs

    refs: list[SkeletonRef] = []
    seen_ids: set[str] = set()
    for item in payload.steps:
        citations = resolve(item.chunk_ids)
        if not citations:
            continue  # grounding gate: drop ungrounded steps
        step_id = upsert_step(
            pool,
            title=item.title,
            description=item.description,
            tags=item.tags,
            citations=citations,
        )
        if step_id in seen_ids:
            continue  # identical-title proposals collapse to one ref
        seen_ids.add(step_id)
        refs.append(SkeletonRef(id=step_id, requirement=item.requirement))
    return refs


# --- invariant gate --------------------------------------------------------


def _enforce_invariants(
    draft: Skeleton, active: Blueprint | None
) -> tuple[Skeleton, list[str]]:
    """Re-inject any human-owned step a draft skeleton would remove or downgrade.

    Protected = ``required`` or ``invariant`` in the active (resolved) blueprint.
    Such steps are never silently dropped: their reference is restored on the
    draft skeleton and the change is reported as an escalation. The referenced
    step already lives in the pool (it is an active step), so the restored ref
    resolves.

    Returns a *new* Skeleton — the input ``draft`` is never mutated.
    """
    if active is None:
        return draft, []

    patched_refs = [r.model_copy(deep=True) for r in draft.steps]
    by_id = {r.id: r for r in patched_refs}
    notes: list[str] = []
    for prev in active.steps:
        protected = prev.requirement == "required" or prev.invariant
        if not protected:
            continue
        existing = by_id.get(prev.id)
        if existing is None:
            patched_refs.append(
                SkeletonRef(
                    id=prev.id,
                    requirement=prev.requirement,
                    invariant=prev.invariant,
                )
            )
            notes.append(
                f"re-injected protected step removed by draft: {prev.title} ({prev.id})"
            )
        elif existing.requirement != prev.requirement:
            existing.requirement = prev.requirement
            existing.invariant = existing.invariant or prev.invariant
            notes.append(
                f"restored requirement of protected step: {prev.title} ({prev.id})"
            )
    patched = draft.model_copy(update={"steps": patched_refs})
    return patched, notes


# --- job -------------------------------------------------------------------


def _model_name(llm: LLMClient) -> str | None:
    for attr in ("chat_model", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str):
            return value
    return None


def _next_version(active: Blueprint | None) -> str:
    if active is None:
        return "1"
    try:
        return str(int(active.version) + 1)
    except ValueError:
        return f"{active.version}-next"


def _generate_scope(
    scope: str,
    *,
    fingerprint: str,
    llm: LLMClient,
    store: VectorStore,
    bm25_cache: BM25IndexCache,
    model: str | None,
    global_steps: list[BlueprintStep] | None = None,
) -> GenerationOutcome:
    active = drafts.active_blueprint(scope)

    # Idempotency: skip if active (or a pending draft) already reflects this corpus.
    for existing in (active, drafts.get_draft(scope)):
        if (
            existing is not None
            and existing.source == "generated"
            and existing.provenance is not None
            and existing.provenance.corpus_fingerprint == fingerprint
        ):
            return GenerationOutcome(
                scope=scope, status="unchanged", notes=["corpus unchanged since draft"]
            )

    if store.count() == 0:
        return GenerationOutcome(
            scope=scope, status="skipped", notes=["corpus is empty"]
        )

    chunks = hybrid_retrieve(
        question=_scope_query(scope),
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=bm25_cache,
    )
    if not chunks:
        return GenerationOutcome(
            scope=scope, status="skipped", notes=["no grounding evidence retrieved"]
        )

    pool = load_pool()
    refs = _draft_steps(scope, chunks, llm, pool, global_steps)
    if not refs:
        return GenerationOutcome(
            scope=scope, status="skipped", notes=["no grounded steps proposed"]
        )

    draft = Skeleton(
        scope=scope,
        version=_next_version(active),
        source="generated",
        steps=refs,
        provenance=BlueprintProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=model,
        ),
    )

    draft, invariant_notes = _enforce_invariants(draft, active)
    if draft.provenance is not None:
        draft.provenance.notes = invariant_notes

    # Persist the new step records before the draft that references them.
    save_pool(pool)
    drafts.save_draft(draft)

    status: OutcomeStatus
    if invariant_notes:
        status = "escalated"
    elif active is None:
        status = "created"
    else:
        status = "updated"
    return GenerationOutcome(
        scope=scope,
        status=status,
        draft_version=draft.version,
        chunks_retrieved=len(chunks),
        steps_drafted=len(refs),
        model=model,
        notes=invariant_notes,
    )


def generate_blueprints(
    llm: LLMClient,
    store: VectorStore,
    *,
    scopes: list[str] | None = None,
) -> list[GenerationOutcome]:
    """Draft/update blueprints for each scope; write drafts to the review queue."""
    fingerprint = corpus_fingerprint(store)
    bm25_cache = BM25IndexCache()
    model = _model_name(llm)

    outcomes: list[GenerationOutcome] = []
    resolved_scopes = scopes or default_scopes()

    # Generate global first so area scopes can exclude its steps.
    if "global" in resolved_scopes:
        resolved_scopes = ["global"] + [s for s in resolved_scopes if s != "global"]

    global_steps: list[BlueprintStep] | None = None
    for scope in resolved_scopes:
        try:
            outcome = _generate_scope(
                scope,
                fingerprint=fingerprint,
                llm=llm,
                store=store,
                bm25_cache=bm25_cache,
                model=model,
                global_steps=global_steps if scope != "global" else None,
            )
            outcomes.append(outcome)
            # After global is generated, collect its steps for area scopes.
            if scope == "global" and global_steps is None:
                bp = drafts.get_draft("global") or drafts.active_blueprint("global")
                if bp is not None:
                    global_steps = bp.steps
        except GenerationError as exc:
            logger.warning("Generation failed for scope %s: %s", scope, exc)
            outcomes.append(
                GenerationOutcome(scope=scope, status="skipped", notes=[str(exc)])
            )
    return outcomes

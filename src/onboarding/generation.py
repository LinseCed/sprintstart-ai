"""AI-authoring of the mandatory onboarding baseline from the ingested corpus.

A batch, re-runnable job that drafts/updates baselines (``scope: global`` and
``scope: area:<name>``) as ``source: generated`` artifacts. It reuses the
existing retrieval layer (:func:`rag.hybrid.hybrid_retrieve`) and the
``LLMClient`` abstraction — there is no separate ingest or retrieval path.

A baseline is a **selection over the competency graph**: which competencies must
everyone in a scope reach, and to what level. It used to be a list of prose
steps — a second content model running in parallel to the graph — and this job's
work was to write that prose. It is now to *choose*, from the catalog the
backend supplies, and to say why.

The job is deliberately conservative:

* **Grounded** — a competency is only selected when the retrieved corpus shows
  the team actually works this way; the model must cite the evidence that put it
  in the baseline. Ungrounded selections are dropped.
* **Catalog-bound** — only keys present in the supplied catalog may be selected.
  An invented key is discarded here, not silently dropped by the backend.
* **Idempotent** — a baseline records the ``corpus_fingerprint`` it was drafted
  from. Re-running against an unchanged corpus is a no-op.
* **Invariant-safe** — it may not remove or downgrade a human-owned entry
  (``required`` or ``invariant`` in the active baseline). Such changes are
  blocked: the protected entry is re-injected and the outcome is escalated.

The backend owns baseline persistence; the AI service is stateless and only
returns generated data.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.graph_models import ActiveCompetency
from onboarding.models import Baseline, BaselineCompetency, BlueprintProvenance
from onboarding.scope import GLOBAL, Scope
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

OutcomeStatus = Literal["created", "updated", "unchanged", "escalated", "skipped"]

_TOP_K = 12
_MIN_SCORE = 0.3

# Proficiency ranks, mirroring the backend's ledger levels.
_MIN_LEVEL = 1
_MAX_LEVEL = 4


class GenerationError(Exception):
    """Raised when the LLM output for a scope cannot be parsed/validated."""


class GenerationOutcome(BaseModel):
    scope: str
    status: OutcomeStatus
    blueprint: Baseline | None = None
    draft_version: str | None = None
    chunks_retrieved: int = 0
    competencies_selected: int = 0
    model: str | None = None
    notes: list[str] = Field(default_factory=list)


# --- LLM payload -----------------------------------------------------------


class _GenEntry(BaseModel):
    competency_key: str
    # Absent/null means "the competency's own bar applies", which is what the
    # model should return unless this scope genuinely demands a different depth.
    target_level: int | None = None
    requirement: Literal["required", "recommended"] = "recommended"
    rationale: str = ""
    chunk_ids: list[str] = Field(default_factory=list)


class _GenPayload(BaseModel):
    competencies: list[_GenEntry] = []


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


def default_scopes(active: list[Baseline]) -> list[str]:
    """``global`` plus every area scope present among the active baselines."""
    scopes = {b.scope for b in active}
    scopes.add(GLOBAL)
    return sorted(scopes)


def _scope_label(scope: str) -> str:
    parsed = Scope.parse(scope)
    if parsed.is_global:
        return "everyone on the team, regardless of working area"
    if parsed.area is not None:
        return f"new team members working in {parsed.area}"
    return scope


def _scope_query(scope: str) -> str:
    parsed = Scope.parse(scope)
    if parsed.is_global:
        return "team onboarding essentials getting started setup access"
    if parsed.area is not None:
        return f"{parsed.area} onboarding essentials setup workflow"
    return f"{scope} onboarding"


# --- prompt / parsing ------------------------------------------------------


def _build_prompt(
    scope: str,
    chunks: list[ScoredChunk],
    competencies: list[ActiveCompetency],
    global_keys: set[str] | None = None,
) -> list[Message]:
    evidence = "\n".join(f"[{c.id}] ({c.filename}) {c.text}" for c in chunks)
    catalog = "\n".join(
        f"- {c.key}: {c.label}" + (f" — {c.description}" if c.description else "")
        for c in competencies
    )
    exclusion = ""
    if global_keys:
        covered = "\n".join(f"- {key}" for key in sorted(global_keys))
        exclusion = (
            "5. ALREADY IN THE GLOBAL BASELINE — the competencies below are "
            "already required of everyone on the team. Do NOT select them again "
            "here; this scope's baseline adds only what is specific to it.\n\n"
            f"{covered}\n\n"
        )
    system = (
        "You choose the mandatory onboarding baseline for a software team: which "
        "competencies from the team's competency graph every new joiner in a "
        "given scope must reach, and how deeply.\n\n"
        "You are given the team's competency catalog and evidence snippets from "
        "its knowledge base, each prefixed with its chunk id in square "
        "brackets.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences). Select a "
        f"concise baseline for {_scope_label(scope)}. Rules:\n"
        "1. Choose ONLY keys from the catalog below. Copy each key EXACTLY as "
        "written; never invent, alter, or approximate a key.\n"
        "2. Every selection MUST cite at least one chunk id showing this team "
        "actually works this way. A competency you cannot ground in the evidence "
        "does not belong in the baseline, however plausible it sounds.\n"
        "3. Select what a new joiner genuinely cannot contribute without. A "
        "baseline everyone must clear is not a wish list: every entry delays the "
        "first contribution of every hire in the scope.\n"
        "4. 'requirement': 'required' for what is genuinely mandatory, "
        "'recommended' otherwise. 'target_level' is 1-4 (1 beginner, 2 "
        "intermediate, 3 advanced, 4 expert) — set it ONLY when this scope needs "
        "a different depth than the competency's own default; otherwise omit it "
        "or use null.\n"
        f"{exclusion}"
        f"\nCompetency catalog (choose from these only):\n{catalog}\n\n"
        'JSON schema: {"competencies": [{"competency_key": str, '
        '"target_level": int|null, "requirement": "required"|"recommended", '
        '"rationale": str, "chunk_ids": [str]}]}'
    )
    user = f"Scope: {scope}\n\nEvidence:\n{evidence}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _select_competencies(
    scope: str,
    chunks: list[ScoredChunk],
    llm: LLMClient,
    competencies: list[ActiveCompetency],
    global_keys: set[str] | None = None,
) -> list[BaselineCompetency]:
    """Ask the LLM to select the baseline; keep the valid, grounded entries.

    Deduplication is exact rather than semantic: entries are competency keys, so
    two selections of the same competency *are* the same selection — the
    embedding-similarity pass prose steps needed has nothing left to do.
    """
    raw = llm.generate(_build_prompt(scope, chunks, competencies, global_keys))
    try:
        payload = _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid generation output: {exc}") from exc

    chunks_by_id = {c.id: c for c in chunks}
    valid_keys = {c.key for c in competencies}

    selected: list[BaselineCompetency] = []
    seen: set[str] = set()
    for item in payload.competencies:
        if item.competency_key not in valid_keys:
            logger.info(
                "Dropped selection of unknown competency %r for scope %s",
                item.competency_key,
                scope,
            )
            continue
        if item.competency_key in seen:
            continue
        if not resolve_citations(item.chunk_ids, chunks_by_id):
            continue  # grounding gate: drop ungrounded selections
        seen.add(item.competency_key)
        selected.append(
            BaselineCompetency(
                competency_key=item.competency_key,
                target_level=_valid_level(item.target_level),
                requirement=item.requirement,
                rationale=item.rationale,
            )
        )
    return selected


def _valid_level(level: int | None) -> int | None:
    """Keep a target-level override only when it is a real proficiency rank."""
    if level is None or not _MIN_LEVEL <= level <= _MAX_LEVEL:
        return None
    return level


# --- invariant gate --------------------------------------------------------


def _enforce_invariants(
    draft: Baseline, active: Baseline | None
) -> tuple[Baseline, list[str]]:
    """Re-inject any human-owned entry a draft would remove or downgrade.

    Protected = ``required`` or ``invariant`` in the active baseline. Such
    entries are never silently dropped: they are restored on the draft and the
    change is reported as an escalation.

    Returns a *new* Baseline — the input ``draft`` is never mutated.
    """
    if active is None:
        return draft, []

    patched = [e.model_copy(deep=True) for e in draft.competencies]
    by_key = {e.competency_key: e for e in patched}
    notes: list[str] = []
    for prev in active.competencies:
        protected = prev.requirement == "required" or prev.invariant
        if not protected:
            continue
        existing = by_key.get(prev.competency_key)
        if existing is None:
            patched.append(prev.model_copy(deep=True))
            notes.append(
                "re-injected protected competency removed by draft: "
                f"{prev.competency_key}"
            )
            continue
        if existing.requirement != prev.requirement:
            existing.requirement = prev.requirement
            existing.invariant = existing.invariant or prev.invariant
            notes.append(
                f"restored requirement of protected competency: {prev.competency_key}"
            )
        if _lowers_bar(existing.target_level, prev.target_level):
            existing.target_level = prev.target_level
            notes.append(
                f"restored target level of protected competency: {prev.competency_key}"
            )
    return draft.model_copy(update={"competencies": patched}), notes


def _lowers_bar(drafted: int | None, protected: int | None) -> bool:
    """True when a draft asks less of a protected entry than the active one.

    ``None`` means "the competency's own bar applies", which is not a number to
    compare — dropping an explicit bar back to the default is itself a downgrade.
    """
    if protected is None:
        return False
    return drafted is None or drafted < protected


# --- job -------------------------------------------------------------------


def _next_version(active: Baseline | None) -> str:
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
    competencies: list[ActiveCompetency],
    active: Baseline | None = None,
    global_keys: set[str] | None = None,
) -> GenerationOutcome:
    # Idempotency: skip if the active baseline already reflects this corpus.
    if (
        active is not None
        and active.source == "generated"
        and active.provenance is not None
        and active.provenance.corpus_fingerprint == fingerprint
    ):
        return GenerationOutcome(
            scope=scope, status="unchanged", notes=["corpus unchanged since active"]
        )

    if not competencies:
        # There is nothing to select from. Proposing an empty baseline would read
        # as "this team requires nothing"; the honest answer is that the graph
        # has to exist first.
        return GenerationOutcome(
            scope=scope, status="skipped", notes=["competency graph is empty"]
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
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
    )
    if not chunks:
        return GenerationOutcome(
            scope=scope, status="skipped", notes=["no grounding evidence retrieved"]
        )

    selected = _select_competencies(scope, chunks, llm, competencies, global_keys)
    if not selected:
        return GenerationOutcome(
            scope=scope, status="skipped", notes=["no grounded competencies selected"]
        )

    draft = Baseline(
        scope=scope,
        version=_next_version(active),
        source="generated",
        competencies=selected,
        provenance=BlueprintProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=model,
        ),
    )

    draft, invariant_notes = _enforce_invariants(draft, active)
    if draft.provenance is not None:
        draft.provenance.notes = invariant_notes

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
        blueprint=draft,
        draft_version=draft.version,
        chunks_retrieved=len(chunks),
        competencies_selected=len(draft.competencies),
        model=model,
        notes=invariant_notes,
    )


def generate_blueprints(
    llm: LLMClient,
    store: VectorStore,
    *,
    scopes: list[str] | None = None,
    active: list[Baseline] | None = None,
    competencies: list[ActiveCompetency] | None = None,
) -> list[GenerationOutcome]:
    """Draft/update baselines for each scope; returns data without persisting.

    ``active`` is the set of currently-active baselines owned by the backend.
    They drive idempotency (skip when the corpus fingerprint is unchanged) and
    version numbering; the AI service holds no state of its own.

    ``competencies`` is the backend's live competency graph — the set every
    baseline is selected from. With an empty catalog there is nothing to select,
    so every scope is skipped rather than proposed empty.
    """
    fingerprint = corpus_fingerprint(store)
    bm25_cache = BM25IndexCache()
    model = llm.model_name
    catalog = competencies or []

    active_by_scope = {b.scope: b for b in (active or [])}
    outcomes: list[GenerationOutcome] = []
    resolved_scopes = scopes or default_scopes(active or [])

    # Generate global first so area scopes can exclude what it already requires.
    if GLOBAL in resolved_scopes:
        resolved_scopes = [GLOBAL] + [s for s in resolved_scopes if s != GLOBAL]

    global_keys: set[str] | None = None
    for scope in resolved_scopes:
        try:
            outcome = _generate_scope(
                scope,
                fingerprint=fingerprint,
                llm=llm,
                store=store,
                bm25_cache=bm25_cache,
                model=model,
                competencies=catalog,
                active=active_by_scope.get(scope),
                global_keys=global_keys if scope != GLOBAL else None,
            )
            outcomes.append(outcome)
            # After global is generated, capture its keys for the area scopes.
            if (
                scope == GLOBAL
                and global_keys is None
                and outcome.blueprint is not None
            ):
                global_keys = {e.competency_key for e in outcome.blueprint.competencies}
        except GenerationError as exc:
            logger.warning("Generation failed for scope %s: %s", scope, exc)
            outcomes.append(
                GenerationOutcome(scope=scope, status="skipped", notes=[str(exc)])
            )
    return outcomes

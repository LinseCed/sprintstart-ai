"""Step registry: in-memory pool operations for onboarding content.

Steps are first-class records keyed by a frozen id. A :class:`Skeleton` (an
ordered list of step references) is resolved against the pool to produce the
served :class:`Blueprint` view. The pool is always in-memory during a single
generation call; the backend owns persistence.

This module imports only :mod:`onboarding.models`; the generation job
(:mod:`onboarding.generation`) builds on top of it.
"""

import logging

from onboarding.models import (
    Blueprint,
    BlueprintStep,
    CitationRef,
    Skeleton,
    StepRecord,
    content_id,
)
from onboarding.similarity import OVERLAP_THRESHOLD, text_overlap

logger = logging.getLogger(__name__)


def upsert_step(
    pool: dict[str, StepRecord],
    *,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    citations: list[CitationRef] | None = None,
    audience: list[str] | None = None,
    min_experience: str | None = None,
    competency_key: str | None = None,
) -> str:
    """Add a step to the pool (or reuse an existing one) and return its id.

    The id is the content fingerprint of the title, so an identical-title step
    deduplicates to a single record. An existing record is reused as-is and
    never overwritten — content edits to a known step are an authoring action on
    the pool, not a side effect of generation.

    A near-duplicate title check (Jaccard on the title tokens) is applied before
    creating a new record: if an existing step's title overlaps at or above
    ``OVERLAP_THRESHOLD``, that step's id is returned instead.  This prevents the
    pool from accumulating multiple entries for the same concept (e.g. "Run
    Service Locally" and "Run the Service Locally").
    """
    step_id = content_id(title)
    if step_id in pool:
        return step_id

    for existing_id, record in pool.items():
        if text_overlap(title, record.title) >= OVERLAP_THRESHOLD:
            logger.debug(
                "upsert_step: collapsed %r into existing step %r (%s)",
                title,
                record.title,
                existing_id,
            )
            return existing_id

    pool[step_id] = StepRecord(
        id=step_id,
        title=title,
        description=description,
        tags=tags or [],
        citations=citations or [],
        audience=audience or [],
        min_experience=min_experience,
        competency_key=competency_key,
    )
    return step_id


def resolve(skeleton: Skeleton, pool: dict[str, StepRecord]) -> Blueprint:
    """Join a skeleton's ordered refs against the pool into a served Blueprint.

    Status (``requirement`` / ``invariant``) comes from the ref; content comes
    from the pooled record. A ref to a step missing from the pool is skipped and
    logged — one dangling ref must not take the serve path down.
    """
    steps: list[BlueprintStep] = []
    for ref in skeleton.steps:
        record = pool.get(ref.id)
        if record is None:
            logger.warning(
                "Skeleton %s references unknown step %s; skipping",
                skeleton.scope,
                ref.id,
            )
            continue
        steps.append(
            BlueprintStep(
                id=record.id,
                title=record.title,
                description=record.description,
                requirement=ref.requirement,
                audience=record.audience,
                min_experience=record.min_experience,
                tags=record.tags,
                resources=record.resources,
                citations=record.citations,
                invariant=ref.invariant,
                competency_key=record.competency_key,
            )
        )
    return Blueprint(
        scope=skeleton.scope,
        version=skeleton.version,
        source=skeleton.source,
        steps=steps,
        provenance=skeleton.provenance,
    )

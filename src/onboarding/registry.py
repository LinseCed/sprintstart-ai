"""Step registry: the storage layer for onboarding content.

Steps are first-class records living in a single pool (``blueprints/steps.yaml``)
keyed by a frozen id. Structure is separate: a :class:`Skeleton` (an on-disk
blueprint file) is an ordered list of references into the pool. Resolving a
skeleton against the pool reconstitutes the served :class:`Blueprint` view, so
the pipeline and management API stay unchanged.

This module imports only :mod:`onboarding.models`; the loader
(:mod:`onboarding.blueprints`) and review queue (:mod:`onboarding.drafts`) build
on top of it.
"""

import logging
import os
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from onboarding.models import (
    Blueprint,
    BlueprintStep,
    CitationRef,
    Skeleton,
    StepRecord,
    content_id,
)

logger = logging.getLogger(__name__)

_POOL_ADAPTER = TypeAdapter(list[StepRecord])

# Repo-root ``blueprints/`` by default; overridable for deployment/tests.
_DEFAULT_BLUEPRINTS_PATH = Path(__file__).resolve().parents[2] / "blueprints"


def blueprints_path() -> Path:
    configured = os.getenv("BLUEPRINTS_PATH", "").strip()
    return Path(configured) if configured else _DEFAULT_BLUEPRINTS_PATH


def _pool_path() -> Path:
    return blueprints_path() / "steps.yaml"


# --- step pool -------------------------------------------------------------


def load_pool() -> dict[str, StepRecord]:
    """Load the step pool as an id-keyed map.

    The pool is a single machine-written file; a malformed pool is logged and
    treated as empty rather than crashing the serve path (file-level resilience,
    consistent with the blueprint loader).
    """
    path = _pool_path()
    if not path.is_file():
        return {}
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        records = _POOL_ADAPTER.validate_python(raw)
    except (yaml.YAMLError, OSError, ValidationError) as exc:
        logger.warning("Failed to read step pool %s: %s", path, exc)
        return {}
    return {record.id: record for record in records}


def save_pool(pool: dict[str, StepRecord]) -> None:
    """Write the pool back as a list sorted by id (stable diffs)."""
    path = _pool_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [pool[key].model_dump(exclude_none=True) for key in sorted(pool)]
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def upsert_step(
    pool: dict[str, StepRecord],
    *,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    citations: list[CitationRef] | None = None,
    audience: list[str] | None = None,
    min_experience: str | None = None,
) -> str:
    """Add a step to the pool (or reuse an existing one) and return its id.

    The id is the content fingerprint of the title, so an identical-title step
    de-duplicates to a single record. An existing record is reused as-is and
    never overwritten — content edits to a known step are an authoring action on
    the pool, not a side effect of generation.
    """
    step_id = content_id(title)
    if step_id not in pool:
        pool[step_id] = StepRecord(
            id=step_id,
            title=title,
            description=description,
            tags=tags or [],
            citations=citations or [],
            audience=audience or [],
            min_experience=min_experience,
        )
    return step_id


# --- resolution ------------------------------------------------------------


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
            )
        )
    return Blueprint(
        scope=skeleton.scope,
        version=skeleton.version,
        source=skeleton.source,
        steps=steps,
        provenance=skeleton.provenance,
    )

"""File-based review queue for AI-proposed blueprints (issue #110).

The generation job never writes the active blueprint directly. It writes a
*draft* to ``blueprints/drafts/``; promotion to active requires explicit human
approval. Approval snapshots the outgoing active blueprint under
``blueprints/versions/<scope>/<version>.yaml`` first, so any version can be
rolled back to. Serving (``onboarding/blueprints.py``) only ever reads the
top-level ``blueprints/*.yaml`` files — drafts and version history live in
sub-directories and are invisible to the serve path.
"""

import logging
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from onboarding.blueprints import load_blueprints
from onboarding.models import Blueprint, Skeleton, StepRecord
from onboarding.registry import blueprints_path, load_pool, resolve
from onboarding.scope import Scope

logger = logging.getLogger(__name__)

ChangeKind = Literal["added", "removed", "modified", "downgraded", "unchanged"]

_SAFE_SCOPE_RE = re.compile(r"^(global|area:[a-z0-9_-]{1,64})$")
_SAFE_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _validate_scope(scope: str) -> str:
    if not _SAFE_SCOPE_RE.match(scope):
        raise ValueError(f"Invalid scope: {scope!r}")
    return scope


def _validate_version(version: str) -> str:
    if not _SAFE_VERSION_RE.match(version):
        raise ValueError(f"Invalid version: {version!r}")
    return version


class StepChange(BaseModel):
    id: str
    change: ChangeKind
    # The change touches a human-owned step (required or invariant in active).
    protected: bool = False


class BlueprintDiff(BaseModel):
    scope: str
    active_version: str | None
    draft_version: str
    changes: list[StepChange] = []
    # True when a protected step would be removed or downgraded.
    blocked: bool = False


# --- paths -----------------------------------------------------------------


def _scope_stem(scope: str) -> str:
    """``global`` -> ``global``; ``area:backend`` -> ``area-backend``.

    Caller must validate ``scope`` before calling this function.
    """
    parsed = Scope.parse(scope)
    if parsed.area is not None:
        return "area-" + parsed.area
    return scope


def _drafts_dir() -> Path:
    return blueprints_path() / "drafts"


def _versions_dir() -> Path:
    return blueprints_path() / "versions"


def _active_path(scope: str) -> Path:
    return blueprints_path() / f"{_scope_stem(scope)}.yaml"


def _draft_path(scope: str) -> Path:
    return _drafts_dir() / f"{_scope_stem(scope)}.yaml"


# --- io helpers ------------------------------------------------------------


def _write(path: Path, skeleton: Skeleton) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = skeleton.model_dump(exclude_none=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _read_skeleton(path: Path) -> Skeleton | None:
    if not path.is_file():
        return None
    try:
        return Skeleton.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError, OSError) as exc:
        logger.warning("Skipping invalid skeleton %s: %s", path, exc)
        return None


def _read_resolved(
    path: Path, pool: dict[str, StepRecord] | None = None
) -> Blueprint | None:
    """Read a skeleton and resolve it against the pool (the served view).

    A caller resolving several skeletons (e.g. :func:`list_drafts`) can pass a
    pre-loaded ``pool`` so the pool file isn't re-read per skeleton.
    """
    skeleton = _read_skeleton(path)
    if skeleton is None:
        return None
    return resolve(skeleton, pool if pool is not None else load_pool())


# --- active blueprints -----------------------------------------------------


def active_blueprint(scope: str) -> Blueprint | None:
    """The currently served blueprint for a scope, if any."""
    _validate_scope(scope)
    for blueprint in load_blueprints():
        if blueprint.scope == scope:
            return blueprint
    return None


# --- draft queue -----------------------------------------------------------


def save_draft(skeleton: Skeleton) -> None:
    _validate_scope(skeleton.scope)
    _write(_draft_path(skeleton.scope), skeleton)


def get_draft(scope: str) -> Blueprint | None:
    _validate_scope(scope)
    return _read_resolved(_draft_path(scope))


def list_drafts() -> list[Blueprint]:
    directory = _drafts_dir()
    if not directory.is_dir():
        return []
    pool = load_pool()
    drafts: list[Blueprint] = []
    for file in sorted(directory.glob("*.yaml")):
        blueprint = _read_resolved(file, pool)
        if blueprint is not None:
            drafts.append(blueprint)
    return drafts


def discard_draft(scope: str) -> bool:
    _validate_scope(scope)
    path = _draft_path(scope)
    if path.is_file():
        path.unlink()
        return True
    return False


# --- version history / rollback --------------------------------------------


def _version_path(scope: str, version: str) -> Path:
    _validate_scope(scope)
    _validate_version(version)
    return _versions_dir() / _scope_stem(scope) / f"{version}.yaml"


def _snapshot(skeleton: Skeleton) -> None:
    """Retain a skeleton version so it can be rolled back to later."""
    _write(_version_path(skeleton.scope, skeleton.version), skeleton)


def list_versions(scope: str) -> list[str]:
    _validate_scope(scope)
    directory = _versions_dir() / _scope_stem(scope)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def get_version(scope: str, version: str) -> Blueprint | None:
    return _read_resolved(_version_path(scope, version))


def _promote(scope: str, skeleton: Skeleton) -> Blueprint:
    """Make ``skeleton`` the active blueprint, snapshotting the outgoing one.

    The current active blueprint (if any) is retained under ``versions/`` first
    so it can be rolled back to.
    """
    current = _read_skeleton(_active_path(scope))
    if current is not None:
        _snapshot(current)

    _write(_active_path(scope), skeleton)
    return resolve(skeleton, load_pool())


def approve_draft(scope: str) -> Blueprint:
    """Promote a draft to active, retaining the outgoing version for rollback."""
    _validate_scope(scope)
    draft = _read_skeleton(_draft_path(scope))
    if draft is None:
        raise FileNotFoundError(f"no draft for scope {scope!r}")

    blueprint = _promote(scope, draft)
    discard_draft(scope)
    return blueprint


def rollback(scope: str, version: str) -> Blueprint:
    """Restore a retained version as the active blueprint."""
    _validate_scope(scope)
    _validate_version(version)
    target = _read_skeleton(_version_path(scope, version))
    if target is None:
        raise FileNotFoundError(f"no version {version!r} for scope {scope!r}")

    return _promote(scope, target)


# --- diff ------------------------------------------------------------------


def diff_against_active(draft: Blueprint) -> BlueprintDiff:
    """Compare a draft against the current active blueprint, by step id.

    A step is *protected* when its active version is ``required`` or
    ``invariant``. Removing or downgrading a protected step sets ``blocked``.
    """
    active = active_blueprint(draft.scope)
    active_steps = {s.id: s for s in active.steps} if active else {}
    draft_steps = {s.id: s for s in draft.steps}

    changes: list[StepChange] = []
    blocked = False

    for step_id, prev in active_steps.items():
        protected = prev.requirement == "required" or prev.invariant
        new = draft_steps.get(step_id)
        if new is None:
            changes.append(
                StepChange(id=step_id, change="removed", protected=protected)
            )
            blocked = blocked or protected
        elif prev.requirement == "required" and new.requirement != "required":
            changes.append(
                StepChange(id=step_id, change="downgraded", protected=protected)
            )
            blocked = blocked or protected
        elif new.model_dump() != prev.model_dump():
            changes.append(
                StepChange(id=step_id, change="modified", protected=protected)
            )

    for step_id in draft_steps:
        if step_id not in active_steps:
            changes.append(StepChange(id=step_id, change="added"))

    return BlueprintDiff(
        scope=draft.scope,
        active_version=active.version if active else None,
        draft_version=draft.version,
        changes=changes,
        blocked=blocked,
    )

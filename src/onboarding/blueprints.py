"""Loading and scope-selection of blueprints.

Authoring is separated from serving: a blueprint file is now a *skeleton*
(ordered references into the step pool, see :mod:`onboarding.registry`) that is
resolved against the pool at load time into the served :class:`Blueprint` view.
Selection is source-agnostic, so the ``generated`` source plugs into the same
seam.
"""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from onboarding.models import Blueprint, PersonProfile, Skeleton
from onboarding.registry import blueprints_path, load_pool, resolve

logger = logging.getLogger(__name__)

# The step pool shares the directory with skeletons but is not itself one.
_POOL_FILENAME = "steps.yaml"


def load_blueprints(path: Path | None = None) -> list[Blueprint]:
    """Load every skeleton under ``path`` and resolve it against the step pool.

    Invalid or unreadable files are skipped and logged rather than crashing the
    serve path — a single bad file must not take the endpoint down.
    """
    directory = path or blueprints_path()
    if not directory.is_dir():
        logger.warning("Blueprints directory does not exist: %s", directory)
        return []

    pool = load_pool()
    blueprints: list[Blueprint] = []
    for file in sorted(directory.glob("*.yaml")):
        if file.name == _POOL_FILENAME:
            continue
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8"))
            blueprints.append(resolve(Skeleton.model_validate(raw), pool))
        except (yaml.YAMLError, ValidationError, OSError) as exc:
            logger.warning("Skipping invalid blueprint %s: %s", file.name, exc)

    return blueprints


def select_blueprints(
    blueprints: list[Blueprint], profile: PersonProfile
) -> list[Blueprint]:
    """Keep ``global`` blueprints and those matching the profile's area.

    An unknown working area yields a global-only path (no matching area scope).
    Global blueprints are ordered first.
    """
    area_scope = f"area:{profile.working_area.strip().lower()}"
    selected = [b for b in blueprints if b.scope == "global"]
    selected.extend(b for b in blueprints if b.scope == area_scope)
    return selected

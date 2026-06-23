"""Loading and scope-selection of blueprints.

Authoring is separated from serving: blueprints are pure YAML data that can be
added or edited without code changes. Selection is source-agnostic, so the
future ``generated`` source (issue #110) plugs into the same seam.
"""

import logging
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from onboarding.models import Blueprint, PersonProfile

logger = logging.getLogger(__name__)

# Repo-root ``blueprints/`` by default; overridable for deployment/tests.
_DEFAULT_BLUEPRINTS_PATH = Path(__file__).resolve().parents[2] / "blueprints"


def blueprints_path() -> Path:
    configured = os.getenv("BLUEPRINTS_PATH", "").strip()
    return Path(configured) if configured else _DEFAULT_BLUEPRINTS_PATH


def load_blueprints(path: Path | None = None) -> list[Blueprint]:
    """Load and validate every ``*.yaml`` blueprint under ``path``.

    Invalid or unreadable files are skipped and logged rather than crashing the
    serve path — a single bad authored file must not take the endpoint down.
    """
    directory = path or blueprints_path()
    if not directory.is_dir():
        logger.warning("Blueprints directory does not exist: %s", directory)
        return []

    blueprints: list[Blueprint] = []
    for file in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8"))
            blueprints.append(Blueprint.model_validate(raw))
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

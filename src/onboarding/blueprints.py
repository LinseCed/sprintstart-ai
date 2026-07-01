"""Scope-selection of blueprints.

Blueprints are owned and persisted by the backend. The AI service receives them
as request input and is stateless. This module provides the pure selection logic
that picks the right blueprint(s) for a given profile.
"""

import logging

from onboarding.models import Blueprint, PersonProfile

logger = logging.getLogger(__name__)


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

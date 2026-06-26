"""Parsing of the onboarding scope identifier.

A scope is either the global scope (``"global"``) or a named area
(``"area:<name>"``). The string form is the persisted/serialized identity (it
lives on :class:`~onboarding.models.Blueprint` / ``Skeleton``); this value object
centralizes how that string is *interpreted* so the ``area:`` prefix and
delimiter live in one place rather than being re-parsed at each call site.
"""

from dataclasses import dataclass

GLOBAL = "global"
AREA_PREFIX = "area:"


@dataclass(frozen=True)
class Scope:
    """A parsed scope identifier."""

    raw: str
    # The area name for an ``area:<name>`` scope; ``None`` for global or any
    # unrecognized form.
    area: str | None

    @property
    def is_global(self) -> bool:
        return self.raw == GLOBAL

    @classmethod
    def parse(cls, raw: str) -> "Scope":
        if raw.startswith(AREA_PREFIX):
            return cls(raw=raw, area=raw[len(AREA_PREFIX) :])
        return cls(raw=raw, area=None)

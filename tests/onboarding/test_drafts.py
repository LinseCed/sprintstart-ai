from pathlib import Path

import pytest
import yaml

from onboarding import drafts
from onboarding.models import Blueprint, BlueprintStep, Source

_SCOPE = "area:backend"


@pytest.fixture(autouse=True)
def tmp_blueprints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BLUEPRINTS_PATH", str(tmp_path))
    return tmp_path


def _seed_active(tmp_path: Path, blueprint: Blueprint) -> None:
    (tmp_path / "area-backend.yaml").write_text(
        yaml.safe_dump(blueprint.model_dump(exclude_none=True)), encoding="utf-8"
    )


def _bp(
    version: str, steps: list[BlueprintStep], source: Source = "generated"
) -> Blueprint:
    return Blueprint(scope=_SCOPE, version=version, source=source, steps=steps)


def test_diff_flags_protected_downgrade_as_blocked(tmp_path: Path) -> None:
    _seed_active(
        tmp_path,
        _bp(
            "1",
            [BlueprintStep(id="sec", title="Security", requirement="required")],
            source="authored",
        ),
    )
    draft = _bp(
        "2",
        [
            BlueprintStep(id="sec", title="Security", requirement="recommended"),
            BlueprintStep(id="new", title="New step"),
        ],
    )

    diff = drafts.diff_against_active(draft)

    by_id = {c.id: c for c in diff.changes}
    assert by_id["sec"].change == "downgraded"
    assert by_id["sec"].protected is True
    assert by_id["new"].change == "added"
    assert diff.blocked is True


def test_diff_unprotected_changes_not_blocked(tmp_path: Path) -> None:
    _seed_active(
        tmp_path,
        _bp("1", [BlueprintStep(id="tour", title="Tour", requirement="recommended")]),
    )
    draft = _bp("2", [BlueprintStep(id="tour", title="Tour (updated)")])

    diff = drafts.diff_against_active(draft)

    assert diff.blocked is False
    assert diff.changes[0].change == "modified"


def test_approve_activates_and_retains_prior_version(tmp_path: Path) -> None:
    _seed_active(
        tmp_path, _bp("1", [BlueprintStep(id="a", title="A")], source="authored")
    )
    drafts.save_draft(_bp("2", [BlueprintStep(id="b", title="B")]))

    promoted = drafts.approve_draft(_SCOPE)

    assert promoted.version == "2"
    active = drafts.active_blueprint(_SCOPE)
    assert active is not None
    assert active.version == "2"
    assert [s.id for s in active.steps] == ["b"]
    assert drafts.get_draft(_SCOPE) is None  # draft consumed
    assert drafts.list_versions(_SCOPE) == ["1"]  # prior retained


def test_rollback_restores_prior_version(tmp_path: Path) -> None:
    _seed_active(
        tmp_path, _bp("1", [BlueprintStep(id="a", title="A")], source="authored")
    )
    drafts.save_draft(_bp("2", [BlueprintStep(id="b", title="B")]))
    drafts.approve_draft(_SCOPE)  # active now v2, v1 retained

    restored = drafts.rollback(_SCOPE, "1")

    assert restored.version == "1"
    active = drafts.active_blueprint(_SCOPE)
    assert active is not None
    assert [s.id for s in active.steps] == ["a"]
    # The outgoing v2 is retained so the rollback itself is reversible.
    assert set(drafts.list_versions(_SCOPE)) == {"1", "2"}


def test_approve_missing_draft_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        drafts.approve_draft(_SCOPE)

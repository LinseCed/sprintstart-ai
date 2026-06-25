from pathlib import Path

import pytest
import yaml

from onboarding import drafts
from onboarding.models import Blueprint, Skeleton, Source, content_id
from onboarding.registry import load_pool, resolve

_SCOPE = "area:backend"


@pytest.fixture(autouse=True)
def tmp_blueprints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BLUEPRINTS_PATH", str(tmp_path))
    return tmp_path


def _seed_pool(tmp_path: Path, *titles: str) -> None:
    records = [{"id": content_id(t), "title": t} for t in titles]
    (tmp_path / "steps.yaml").write_text(yaml.safe_dump(records), encoding="utf-8")


def _ref(
    title: str, requirement: str = "recommended", invariant: bool = False
) -> dict[str, object]:
    return {"id": content_id(title), "requirement": requirement, "invariant": invariant}


def _skeleton(
    version: str, refs: list[dict[str, object]], source: Source = "generated"
) -> Skeleton:
    return Skeleton.model_validate(
        {"scope": _SCOPE, "version": version, "source": source, "steps": refs}
    )


def _seed_active(tmp_path: Path, skeleton: Skeleton) -> None:
    (tmp_path / "area-backend.yaml").write_text(
        yaml.safe_dump(skeleton.model_dump(exclude_none=True)), encoding="utf-8"
    )


def _resolved(skeleton: Skeleton) -> Blueprint:
    return resolve(skeleton, load_pool())


def test_diff_flags_protected_downgrade_as_blocked(tmp_path: Path) -> None:
    _seed_pool(tmp_path, "Security", "New step")
    _seed_active(
        tmp_path, _skeleton("1", [_ref("Security", "required")], source="authored")
    )
    draft = _resolved(
        _skeleton("2", [_ref("Security", "recommended"), _ref("New step")])
    )

    diff = drafts.diff_against_active(draft)

    sec_id = content_id("Security")
    new_id = content_id("New step")
    by_id = {c.id: c for c in diff.changes}
    assert by_id[sec_id].change == "downgraded"
    assert by_id[sec_id].protected is True
    assert by_id[new_id].change == "added"
    assert diff.blocked is True


def test_diff_unprotected_changes_not_blocked(tmp_path: Path) -> None:
    _seed_pool(tmp_path, "Tour")
    _seed_active(
        tmp_path, _skeleton("1", [_ref("Tour", "recommended")], source="generated")
    )
    # A skeleton draft can only change status/membership (content is pooled);
    # an unprotected recommended->required upgrade is a "modified", not blocked.
    draft = _resolved(_skeleton("2", [_ref("Tour", "required")]))

    diff = drafts.diff_against_active(draft)

    assert diff.blocked is False
    assert diff.changes[0].change == "modified"


def test_approve_activates_and_retains_prior_version(tmp_path: Path) -> None:
    _seed_pool(tmp_path, "A", "B")
    _seed_active(tmp_path, _skeleton("1", [_ref("A")], source="authored"))
    drafts.save_draft(_skeleton("2", [_ref("B")]))

    promoted = drafts.approve_draft(_SCOPE)

    assert promoted.version == "2"
    active = drafts.active_blueprint(_SCOPE)
    assert active is not None
    assert active.version == "2"
    assert [s.id for s in active.steps] == [content_id("B")]
    assert drafts.get_draft(_SCOPE) is None  # draft consumed
    assert drafts.list_versions(_SCOPE) == ["1"]  # prior retained


def test_rollback_restores_prior_version(tmp_path: Path) -> None:
    _seed_pool(tmp_path, "A", "B")
    _seed_active(tmp_path, _skeleton("1", [_ref("A")], source="authored"))
    drafts.save_draft(_skeleton("2", [_ref("B")]))
    drafts.approve_draft(_SCOPE)  # active now v2, v1 retained

    restored = drafts.rollback(_SCOPE, "1")

    assert restored.version == "1"
    active = drafts.active_blueprint(_SCOPE)
    assert active is not None
    assert [s.id for s in active.steps] == [content_id("A")]
    # The outgoing v2 is retained so the rollback itself is reversible.
    assert set(drafts.list_versions(_SCOPE)) == {"1", "2"}


def test_approve_missing_draft_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        drafts.approve_draft(_SCOPE)

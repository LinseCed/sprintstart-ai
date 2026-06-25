from pathlib import Path

import pytest

from onboarding.models import (
    CitationRef,
    Skeleton,
    SkeletonRef,
    StepRecord,
    content_id,
)
from onboarding.registry import load_pool, resolve, save_pool, upsert_step


@pytest.fixture(autouse=True)
def _tmp_blueprints(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setenv("BLUEPRINTS_PATH", str(tmp_path))
    return tmp_path


def _record(title: str, **kwargs: object) -> StepRecord:
    return StepRecord(id=content_id(title), title=title, **kwargs)  # type: ignore[arg-type]


def test_resolve_merges_pool_content_with_ref_status() -> None:
    record = _record(
        "Set up DB",
        tags=["db"],
        citations=[CitationRef(filename="a.md", chunk_id="c1")],
    )
    skel = Skeleton(
        scope="area:backend",
        steps=[SkeletonRef(id=record.id, requirement="required", invariant=True)],
    )

    bp = resolve(skel, {record.id: record})

    assert bp.scope == "area:backend"
    [step] = bp.steps
    assert step.title == "Set up DB"  # content from the pool record
    assert step.tags == ["db"]
    assert step.citations[0].chunk_id == "c1"
    assert step.requirement == "required"  # status from the skeleton ref
    assert step.invariant is True


def test_resolve_skips_dangling_ref() -> None:
    skel = Skeleton(scope="global", steps=[SkeletonRef(id="step-missing")])
    assert resolve(skel, {}).steps == []


def test_frozen_id_survives_rename() -> None:
    record = _record("Set up DB")
    save_pool({record.id: record})
    sid = record.id

    pool = load_pool()
    pool[sid].title = "Provision the database"  # rename in place
    save_pool(pool)
    reloaded = load_pool()

    assert sid in reloaded  # id is frozen, not recomputed from the new title
    assert reloaded[sid].title == "Provision the database"
    assert content_id("Provision the database") != sid


def test_upsert_dedups_by_fingerprint() -> None:
    pool: dict[str, StepRecord] = {}
    first = upsert_step(pool, title="Set up DB", description="first")
    second = upsert_step(pool, title="set up   db", description="second")

    assert first == second  # same normalized title -> same fingerprint
    assert len(pool) == 1
    assert pool[first].description == "first"  # existing record kept, not overwritten

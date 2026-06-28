from onboarding.models import (
    CitationRef,
    Skeleton,
    SkeletonRef,
    StepRecord,
    content_id,
)
from onboarding.registry import resolve, upsert_step


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


def test_upsert_dedups_by_fingerprint() -> None:
    pool: dict[str, StepRecord] = {}
    first = upsert_step(pool, title="Set up DB", description="first")
    second = upsert_step(pool, title="set up   db", description="second")

    assert first == second  # same normalized title -> same fingerprint
    assert len(pool) == 1
    assert pool[first].description == "first"  # existing record kept, not overwritten

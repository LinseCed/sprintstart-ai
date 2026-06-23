# pyright: reportPrivateUsage=false
# Unit-tests the pipeline's gate internals directly (coverage/invariants/filter).
from onboarding.models import (
    Blueprint,
    BlueprintStep,
    CitationRef,
    OnboardingPath,
    PathPhase,
    PathStep,
    PersonProfile,
    QualityReport,
    experience_rank,
)
from onboarding.pipeline import (
    _enforce_coverage,
    _enforce_invariants,
    _step_applies,
)
from onboarding.quality import evaluate


def _path(phases: list[PathPhase]) -> OnboardingPath:
    return OnboardingPath(
        working_area="backend",
        experience="junior",
        phases=phases,
        quality=QualityReport(
            coverage=0.0, grounded_ratio=0.0, ordering_valid=False, score=0.0
        ),
    )


def test_experience_rank_unknown_is_zero() -> None:
    assert experience_rank("junior") == 1
    assert experience_rank("astronaut") == 0
    assert experience_rank(None) == 0


def test_required_step_always_applies_regardless_of_experience() -> None:
    step = BlueprintStep(
        id="x", title="X", requirement="required", min_experience="senior"
    )
    profile = PersonProfile(working_area="backend", experience="junior")

    assert _step_applies(step, profile) is True


def test_recommended_step_filtered_by_min_experience() -> None:
    step = BlueprintStep(
        id="x", title="X", requirement="recommended", min_experience="senior"
    )
    junior = PersonProfile(working_area="backend", experience="junior")
    senior = PersonProfile(working_area="backend", experience="senior")

    assert _step_applies(step, junior) is False
    assert _step_applies(step, senior) is True


def test_recommended_step_filtered_by_audience() -> None:
    step = BlueprintStep(
        id="x", title="X", requirement="recommended", audience=["frontend"]
    )
    backend = PersonProfile(working_area="backend", experience="junior")
    assert _step_applies(step, backend) is False

    tagged = PersonProfile(
        working_area="backend", experience="junior", tags=["frontend"]
    )
    assert _step_applies(step, tagged) is True


def test_coverage_gate_reinjects_missing_required_step() -> None:
    blueprints = [
        Blueprint(
            scope="global",
            steps=[
                BlueprintStep(id="sec", title="Security", requirement="required"),
                BlueprintStep(id="acc", title="Accounts", requirement="required"),
            ],
        )
    ]
    # Path is missing the required "acc" step.
    phases = [
        PathPhase(
            title="Getting started",
            steps=[PathStep(id="sec", title="Security", requirement="required")],
        )
    ]
    profile = PersonProfile(working_area="backend", experience="junior")

    _enforce_coverage(phases, blueprints, profile)

    present = {s.id for p in phases for s in p.steps}
    assert present == {"sec", "acc"}


def test_invariants_gate_adds_security_step_when_absent() -> None:
    phases = [
        PathPhase(
            title="Getting started",
            steps=[PathStep(id="acc", title="Accounts", requirement="required")],
        )
    ]

    _enforce_invariants(phases)

    ids = [s.id for p in phases for s in p.steps]
    assert "security-policy-ack" in ids


def test_quality_rubric_scores_perfect_path() -> None:
    phases = [
        PathPhase(
            title="Getting started",
            steps=[
                PathStep(id="sec", title="Security", requirement="required"),
                PathStep(
                    id="llm-1",
                    title="Extra",
                    requirement="recommended",
                    origin="llm",
                    citations=[CitationRef(filename="a.md", chunk_id="c1")],
                ),
            ],
        )
    ]
    path = _path(phases)

    report = evaluate(path, required_step_ids={"sec"})

    assert report.coverage == 1.0
    assert report.grounded_ratio == 1.0
    assert report.ordering_valid is True
    assert report.score == 1.0


def test_quality_rubric_flags_missing_coverage_and_bad_ordering() -> None:
    phases = [
        PathPhase(
            title="Getting started",
            steps=[
                PathStep(id="rec", title="Recommended", requirement="recommended"),
                PathStep(id="req", title="Required", requirement="required"),
            ],
        )
    ]
    path = _path(phases)

    report = evaluate(path, required_step_ids={"req", "missing"})

    assert report.coverage == 0.5
    assert report.ordering_valid is False
    assert any("missing required" in n for n in report.notes)


def test_path_serializes_to_yaml_round_trip() -> None:
    import yaml

    path = _path([PathPhase(title="P", steps=[PathStep(id="a", title="A")])])

    loaded = yaml.safe_load(path.to_yaml())

    assert loaded["working_area"] == "backend"
    assert loaded["phases"][0]["steps"][0]["id"] == "a"

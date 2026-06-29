"""Deterministic quality rubric over an assembled onboarding path.

No LLM is involved, so the score is reproducible and regression-testable. An
LLM-judge could later be layered behind this without changing the gate.
"""

from onboarding.models import OnboardingPath, QualityReport

# Weights for the composite score; sum to 1.0.
_W_COVERAGE = 0.6
_W_GROUNDING = 0.3
_W_ORDERING = 0.1


def evaluate(
    path: OnboardingPath,
    required_step_ids: set[str],
    extra_notes: list[str] | None = None,
) -> QualityReport:
    """Score the path on coverage, grounding, and ordering.

    ``required_step_ids`` are the ids of every required step expected for the
    person's scope (the coverage gate guarantees they are present).
    """
    steps = [step for phase in path.phases for step in phase.steps]
    present_ids = {step.id for step in steps}

    coverage = (
        len(required_step_ids & present_ids) / len(required_step_ids)
        if required_step_ids
        else 1.0
    )

    llm_steps = [s for s in steps if s.origin == "llm"]
    grounded = [s for s in llm_steps if s.citations]
    grounded_ratio = len(grounded) / len(llm_steps) if llm_steps else 1.0

    ordering_valid = _ordering_valid(path)

    score = (
        _W_COVERAGE * coverage
        + _W_GROUNDING * grounded_ratio
        + _W_ORDERING * (1.0 if ordering_valid else 0.0)
    )

    notes = list(extra_notes or [])
    missing = required_step_ids - present_ids
    if missing:
        notes.append(f"missing required steps: {sorted(missing)}")
    if not ordering_valid:
        notes.append("required steps appear after recommended steps")

    return QualityReport(
        coverage=round(coverage, 3),
        grounded_ratio=round(grounded_ratio, 3),
        ordering_valid=ordering_valid,
        score=round(score, 3),
        notes=notes,
    )


def _ordering_valid(path: OnboardingPath) -> bool:
    """Within each phase, no required step may follow a recommended one."""
    for phase in path.phases:
        seen_recommended = False
        for step in phase.steps:
            if step.requirement == "recommended":
                seen_recommended = True
            elif seen_recommended:
                return False
    return True

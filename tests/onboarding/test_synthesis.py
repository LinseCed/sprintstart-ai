# pyright: reportPrivateUsage=false
# Unit-tests the synthesis prompt builder: skills/interests must reach the LLM.
from onboarding.models import BlueprintStep, PersonProfile
from onboarding.synthesis import _build_prompt


def test_synthesis_prompt_includes_skills_and_interests() -> None:
    profile = PersonProfile(
        working_area="backend",
        experience="junior",
        skills=["python", "fastapi"],
        tags=["testing"],
    )
    steps = [BlueprintStep(id="x", title="Setup", requirement="required")]

    messages = _build_prompt(profile, steps, {})

    user = next(m["content"] for m in messages if m["role"] == "user")
    assert "python" in user
    assert "fastapi" in user
    assert "testing" in user

"""The mentor's persona is assembled per hire, not fixed.

The through-line: **the persona must never describe a capability this hire does not
have.** A mentor told about a tool it was not given will offer the hire something
impossible, and a mentor told to celebrate merges will say so to a Scrum Master.
"""

from onboarding.buddy_persona import build_persona
from onboarding.vocabulary import DEFAULT_VOCABULARY, Vocabulary

_ALL_TOOLS = (
    "search_docs",
    "get_learning_plan",
    "get_module",
    "get_my_metrics",
    "get_my_competencies",
    "get_suggested_tasks",
    "submit_verification",
    "claim_goal",
    "flag_to_pm",
)


def test_full_toolset_persona_mentions_every_mounted_tool() -> None:
    persona = build_persona(_ALL_TOOLS)

    for tool in _ALL_TOOLS:
        assert f"`{tool}`" in persona


def test_a_tool_that_is_not_mounted_is_never_mentioned() -> None:
    mounted = [t for t in _ALL_TOOLS if t != "get_module"]

    persona = build_persona(mounted)

    # Not softened to "if available" -- absent, so the model cannot try to call it.
    assert "`get_module`" not in persona
    assert "`get_learning_plan`" in persona


def test_without_an_escalation_tool_the_persona_offers_no_escalation() -> None:
    persona = build_persona([t for t in _ALL_TOOLS if t != "flag_to_pm"])

    assert "flag_to_pm" not in persona
    # The honesty instruction survives; only the offer of a route out goes.
    assert "rather than inventing an answer" in persona


def test_hire_state_tools_are_listed_only_when_mounted() -> None:
    persona = build_persona(["search_docs", "get_my_metrics"])

    assert "`get_my_metrics`" in persona
    assert "`get_my_competencies`" not in persona
    assert "`get_suggested_tasks`" not in persona


def test_no_hire_state_tools_means_no_hire_state_clause() -> None:
    persona = build_persona(["search_docs"])

    assert "hire-state tools" not in persona


def test_default_vocabulary_is_the_engineering_wording() -> None:
    persona = build_persona(_ALL_TOOLS, DEFAULT_VOCABULARY)

    assert "Celebrate the changes and milestones" in persona


def test_a_tracks_vocabulary_replaces_the_engineering_wording() -> None:
    delivery = Vocabulary(
        contribution_noun="ceremony",
        contribution_noun_plural="ceremonies",
        contribution_verb_past="facilitated",
    )

    persona = build_persona(_ALL_TOOLS, delivery)

    assert "Celebrate the ceremonies and milestones" in persona
    assert "changes" not in persona


def test_the_persona_never_presumes_the_hire_writes_code() -> None:
    persona = build_persona(
        [t for t in _ALL_TOOLS if t != "get_my_metrics"],
        Vocabulary("plan", "plans", "published"),
    )

    # The regression this whole slice exists for: a hire whose work is never a pull
    # request must not meet a mentor whose standing instructions are about merging.
    lowered = persona.lower()
    assert "pull request" not in lowered
    assert "merge" not in lowered


def test_the_grounding_rule_survives_every_toolset() -> None:
    for mounted in ([], ["search_docs"], _ALL_TOOLS):
        persona = build_persona(mounted)

        # Non-negotiable regardless of what is mounted: grounding and the
        # test/fixture caveat are safety rules, not capabilities.
        assert "Ground every claim" in persona
        assert "test, fixture, or sample-data files" in persona

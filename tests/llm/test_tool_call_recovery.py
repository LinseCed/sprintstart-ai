"""Tests for recovering tool calls a model leaked as text markup."""

from llm.tool_call_recovery import recover_tool_calls

# The exact shape a hire saw leak into a buddy answer: DeepSeek's tool-call markup
# (fullwidth-pipe DSML delimiters) that OpenRouter did not lift into tool_calls.
_LEAKED = (
    '<｜｜DSML｜｜tool_calls> <｜｜DSML｜｜invoke name="search_docs"> '
    '<｜｜DSML｜｜parameter name="query" string="true">'
    "open pull request waiting review 1514 hours"
    "</｜｜DSML｜｜parameter> </｜｜DSML｜｜invoke> </｜｜DSML｜｜tool_calls>"
)


def test_recovers_a_leaked_search_docs_call():
    calls, text = recover_tool_calls(_LEAKED)

    assert len(calls) == 1
    assert calls[0].name == "search_docs"
    assert calls[0].arguments == {
        "query": "open pull request waiting review 1514 hours"
    }
    # The markup is stripped; nothing of it is left as visible text.
    assert text == ""


def test_keeps_prose_before_the_leaked_block():
    content = "Let me check that for you.\n\n" + _LEAKED
    calls, text = recover_tool_calls(content)

    assert len(calls) == 1
    assert text == "Let me check that for you."


def test_recovers_multiple_invokes():
    content = (
        '<invoke name="get_my_metrics"></invoke>'
        '<invoke name="search_docs">'
        '<parameter name="query">code review SLA</parameter></invoke>'
    )
    calls, _ = recover_tool_calls(content)

    assert [call.name for call in calls] == ["get_my_metrics", "search_docs"]
    assert calls[1].arguments == {"query": "code review SLA"}


def test_plain_text_answer_is_left_untouched():
    content = "You have two open pull requests: #128 and #142."
    calls, text = recover_tool_calls(content)

    assert calls == []
    assert text == content


def test_prose_mentioning_the_word_invoke_is_not_a_false_positive():
    # No markup corroborates it, so it must not be parsed as a call.
    content = "To invoke the build you run ./gradlew build."
    calls, text = recover_tool_calls(content)

    assert calls == []
    assert text == content


def test_coerces_non_string_parameter_values():
    content = (
        '<invoke name="submit_verification">'
        '<parameter name="passed">true</parameter>'
        '<parameter name="count">3</parameter></invoke>'
    )
    calls, _ = recover_tool_calls(content)

    assert calls[0].arguments == {"passed": True, "count": 3}

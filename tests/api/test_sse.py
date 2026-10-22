import json
import logging
from collections.abc import Iterator

import pytest

from api.sse import stream_events
from llm.errors import LLMUnavailableError


def _decode(frames: list[str]) -> list[dict[str, object]]:
    return [json.loads(frame.removeprefix("data: ")) for frame in frames]


def test_events_are_framed_one_per_data_line_in_order() -> None:
    frames = list(stream_events([{"type": "a"}, {"type": "b", "n": 1}], what="x"))

    assert frames == ['data: {"type": "a"}\n\n', 'data: {"type": "b", "n": 1}\n\n']


def test_an_unavailable_model_ends_the_stream_with_its_message() -> None:
    def events() -> Iterator[dict[str, object]]:
        yield {"type": "token", "content": "hi"}
        raise LLMUnavailableError("model offline")

    assert _decode(list(stream_events(events(), what="x"))) == [
        {"type": "token", "content": "hi"},
        {"type": "error", "message": str(LLMUnavailableError("model offline"))},
    ]


def test_any_other_failure_ends_the_stream_with_a_generic_message_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def events() -> Iterator[dict[str, object]]:
        yield {"type": "token", "content": "hi"}
        raise RuntimeError("secret internals")

    with caplog.at_level(logging.ERROR):
        frames = list(stream_events(events(), what="the thing stream"))

    assert _decode(frames) == [
        {"type": "token", "content": "hi"},
        {"type": "error", "message": "An unexpected error occurred"},
    ]
    assert "secret internals" not in "".join(frames)
    assert [r.getMessage() for r in caplog.records] == [
        "Unexpected error in the thing stream"
    ]

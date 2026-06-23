"""SSE orchestrator for onboarding-path generation.

Mirrors :class:`agents.orchestrator.ChatOrchestrator`: a generator that streams
Server-Sent Events. Path generation is multi-step, so each pipeline stage is
emitted as a ``stage`` event, followed by a single ``path`` event (structured
path + YAML + quality report) and a ``done`` event. Errors collapse to one
``error`` event, consistent with the chat endpoint.
"""

import json
import logging
from collections.abc import Generator, Iterator

from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.models import OnboardingPath, PersonProfile
from onboarding.pipeline import OnboardingPipeline, StageProgress
from store.base import VectorStore

logger = logging.getLogger(__name__)


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _emit_stages(
    gen: Generator[StageProgress, None, OnboardingPath],
) -> Generator[str, None, OnboardingPath]:
    while True:
        try:
            stage = next(gen)
        except StopIteration as stop:
            return stop.value
        yield _sse({"type": "stage", "name": stage.name})


class OnboardingOrchestrator:
    def __init__(self, llm: LLMClient, store: VectorStore) -> None:
        self._pipeline = OnboardingPipeline(llm, store)

    def stream(self, profile: PersonProfile) -> Iterator[str]:
        try:
            path = yield from _emit_stages(self._pipeline.run(profile))

            yield _sse(
                {
                    "type": "path",
                    "path": path.model_dump(),
                    "path_yaml": path.to_yaml(),
                    "quality": path.quality.model_dump(),
                }
            )
            yield _sse({"type": "done"})

        except LLMUnavailableError as exc:
            yield _sse({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in onboarding stream")
            yield _sse({"type": "error", "message": "An unexpected error occurred"})

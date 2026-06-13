from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field, ValidationError

from agents.tools.base import Delegation, Invocation, Tool, ToolResult
from rag.types import ScoredChunk

if TYPE_CHECKING:
    from agents.base import AgentResult, AgentRunState

_SUMMARY_FILES = 5


def _delegation_summary(name: str, chunks: list[ScoredChunk]) -> str:
    if not chunks:
        return f"{name}: gathered no sources."
    files = sorted({c.filename for c in chunks})
    shown = ", ".join(files[:_SUMMARY_FILES])
    more = "" if len(files) <= _SUMMARY_FILES else ", …"
    return (
        f"{name}: gathered {len(chunks)} chunk(s) from {shown}{more} "
        "and prepared an answer."
    )


class SubAgent(Protocol):
    name: str
    description: str

    def run(self, task: str) -> "AgentResult": ...

    def gather_stream(
        self, task: str
    ) -> "Generator[Invocation, None, AgentRunState]": ...

    def answer_stream(self, task: str, state: "AgentRunState") -> Iterator[str]: ...


class AgentTaskArgs(BaseModel):
    task: str = Field(description="A self-contained instruction for the sub-agent.")


class AgentTool(Tool[AgentTaskArgs]):
    args_model = AgentTaskArgs
    kind = "agent"

    def __init__(self, agent: SubAgent) -> None:
        self._agent = agent
        self.name = agent.name
        self.description = agent.description

    def run(self, args: AgentTaskArgs) -> ToolResult:
        result = self._agent.run(args.task)
        return ToolResult(
            summary=result.answer,
            chunks=result.chunks,
            usages=result.usages,
        )

    def stream(
        self, raw_args: dict[str, object]
    ) -> Generator[Invocation, None, ToolResult]:
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError:
            return ToolResult.empty(f"Invalid arguments for tool {self.name!r}.")

        agent = self._agent
        task = args.task
        state = yield from agent.gather_stream(task)
        delegation = Delegation(
            name=agent.name,
            answer=lambda: agent.answer_stream(task, state),
        )
        return ToolResult(
            summary=_delegation_summary(agent.name, state.chunks),
            chunks=state.chunks,
            usages=state.usages,
            delegation=delegation,
        )

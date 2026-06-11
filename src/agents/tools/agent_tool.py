from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ValidationError

from agents.tools.base import Invocation, Tool, ToolResult

if TYPE_CHECKING:
    from agents.base import AgentResult, AgentRunState


class SubAgent(Protocol):
    name: str
    description: str

    def run(self, task: str) -> "AgentResult": ...

    def gather_stream(
        self, task: str
    ) -> "Generator[Invocation, None, AgentRunState]": ...

    def answer_stream(self, task: str, state: "AgentRunState") -> Iterator[str]: ...


class AgentTaskArgs(BaseModel):
    task: str


class AgentTool(Tool[AgentTaskArgs]):
    args_schema = '{"task": "a self-contained instruction for the sub-agent"}'
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

        state = yield from self._agent.gather_stream(args.task)
        answer = "".join(self._agent.answer_stream(args.task, state))
        return ToolResult(summary=answer, chunks=state.chunks, usages=state.usages)

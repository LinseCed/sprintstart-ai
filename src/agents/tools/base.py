from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from rag.types import ScoredChunk

# Whether an invoked capability is a leaf tool or another (sub-)agent.
CapabilityKind = Literal["agent", "tool"]


@dataclass(frozen=True)
class Invocation:
    kind: CapabilityKind
    name: str


@dataclass(frozen=True)
class ToolResult:
    summary: str
    chunks: list[ScoredChunk] = field(default_factory=list[ScoredChunk])
    usages: list[Invocation] = field(default_factory=list[Invocation])

    @classmethod
    def empty(cls, summary: str) -> "ToolResult":
        return cls(summary=summary, chunks=[])


class Tool[ArgsT: BaseModel](ABC):
    name: str
    description: str
    args_schema: str
    args_model: type[ArgsT]
    kind: CapabilityKind = "tool"

    def execute(self, raw_args: dict[str, object]) -> ToolResult:
        """Validate raw LLM-supplied arguments, then run. Never raises."""
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError:
            return ToolResult.empty(f"Invalid arguments for tool {self.name!r}.")
        return self.run(args)

    @abstractmethod
    def run(self, args: ArgsT) -> ToolResult: ...


@runtime_checkable
class StreamingTool(Protocol):
    def stream(
        self, raw_args: dict[str, object]
    ) -> Generator[Invocation, None, ToolResult]: ...


# Tools are stored heterogeneously; their concrete arg models differ, so the
# registry treats them as `Tool[Any]` to sidestep generic invariance.
AnyTool = Tool[Any]


class ToolRegistry:
    def __init__(self, tools: Iterator[AnyTool] | list[AnyTool]) -> None:
        self._tools: dict[str, AnyTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AnyTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AnyTool | None:
        return self._tools.get(name)

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def execute(self, name: str, raw_args: dict[str, object]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.empty(f"Unknown tool: {name!r}.")
        return tool.execute(raw_args)

    def render(self) -> str:
        return "\n".join(
            f"  {tool.name} — {tool.args_schema}\n      {tool.description}"
            for tool in self._tools.values()
        )

    def __len__(self) -> int:
        return len(self._tools)

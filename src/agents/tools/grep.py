from pydantic import BaseModel, field_validator

from agents.tools.base import Tool, ToolResult
from rag.types import ScoredChunk
from store.base import VectorStore

_GREP_SCORE = 1.0


class GrepArgs(BaseModel):
    patterns: list[str]

    @field_validator("patterns", mode="before")
    @classmethod
    def coerce_to_list(cls, v: object) -> object:
        return [v] if isinstance(v, str) else v


class GrepTool(Tool[GrepArgs]):
    name = "grep"
    description = (
        "Exact (case-insensitive) substring search for identifiers or string "
        "literals. Use when you know a function name, symbol or exact phrase."
    )
    args_model = GrepArgs

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, args: GrepArgs) -> ToolResult:
        needles = [p.lower() for p in args.patterns]
        results = [
            ScoredChunk(
                id=chunk.id,
                artifact_id=chunk.artifact_id,
                filename=chunk.filename,
                text=chunk.text,
                score=_GREP_SCORE,
                position=chunk.position,
                kind=chunk.kind,
            )
            for chunk in self._store.all_chunks_without_embeddings()
            if any(needle in chunk.text.lower() for needle in needles)
        ]
        return ToolResult(
            summary=f"grep({args.patterns}): {len(results)} match(es).",
            chunks=results,
        )

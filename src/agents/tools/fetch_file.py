import re

from pydantic import BaseModel

from agents.tools.base import Tool, ToolResult
from rag.types import ScoredChunk
from store.base import VectorStore

_FETCH_SCORE = 0.9
_EXTENSION_RE = re.compile(r"\.[a-z0-9]+$")


def _stem(filename: str) -> str:
    return _EXTENSION_RE.sub("", filename.lower())


class FetchFileArgs(BaseModel):
    filename: str


class FetchFileTool(Tool[FetchFileArgs]):
    name = "fetch_file"
    description = (
        "Return all indexed chunks from a specific file. "
        "Use when the relevant filename is already known."
    )
    args_schema = '{"filename": "example.py"}'
    args_model = FetchFileArgs

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, args: FetchFileArgs) -> ToolResult:
        target = args.filename.lower()
        target_stem = _stem(args.filename)
        results = [
            ScoredChunk(
                id=chunk.id,
                artifact_id=chunk.artifact_id,
                filename=chunk.filename,
                text=chunk.text,
                score=_FETCH_SCORE,
                heading_path=chunk.heading_path,
                position=chunk.position,
                kind=chunk.kind,
            )
            for chunk in self._store.all_chunks()
            if chunk.filename.lower() == target or _stem(chunk.filename) == target_stem
        ]
        return ToolResult(
            summary=f"fetch_file({args.filename!r}): {len(results)} chunk(s).",
            chunks=results,
        )

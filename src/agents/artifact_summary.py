from dataclasses import dataclass

from llm.base import LLMClient, Message
from rag.types import Chunk

_MAX_BATCH_CHARS = 10_000


@dataclass(frozen=True)
class SummaryCitation:
    artifact_id: str
    filename: str
    source_url: str


@dataclass(frozen=True)
class ArtifactSummary:
    artifact_id: str
    summary: str
    citations: list[SummaryCitation]


class ArtifactSummaryAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def summarize(
        self,
        artifact_id: str,
        chunks: list[Chunk],
        previous_chunks: list[Chunk] | None = None,
    ) -> ArtifactSummary:
        if not chunks:
            raise ValueError("Cannot summarize artifact without chunks.")

        current_notes = self._notes_for_chunks("current artifact", chunks)
        previous_notes = (
            self._notes_for_chunks("previous artifact", previous_chunks)
            if previous_chunks
            else None
        )

        summary = self._final_summary(
            artifact_id=artifact_id,
            current_notes=current_notes,
            previous_notes=previous_notes,
        )

        citations = [_citation_for_chunks(artifact_id, chunks)]

        if previous_chunks:
            citations.append(
                _citation_for_chunks(previous_chunks[0].artifact_id, previous_chunks)
            )

        return ArtifactSummary(
            artifact_id=artifact_id,
            summary=summary,
            citations=citations,
        )

    def _notes_for_chunks(self, label: str, chunks: list[Chunk]) -> str:
        batches = _chunk_batches(chunks, _MAX_BATCH_CHARS)

        if len(batches) == 1:
            return batches[0]

        partials: list[str] = []

        for index, batch in enumerate(batches, start=1):
            messages = _grounded_summary_messages(
                user_content=(
                    f"Summarize batch {index} of the {label}.\n\n"
                    "Extract only grounded facts:\n"
                    "- key points\n"
                    "- decisions\n"
                    "- changes or version-related notes\n\n"
                    f"Source excerpts:\n{batch}"
                )
            )
            partials.append(self._llm.generate(messages).strip())

        return "\n\n".join(partials)

    def _final_summary(
        self,
        artifact_id: str,
        current_notes: str,
        previous_notes: str | None,
    ) -> str:
        previous_section = (
            f"\n\nPrevious version excerpts / notes:\n{previous_notes}\n"
            if previous_notes
            else (
                "\n\nNo previous artifact was provided. If the current source "
                "contains version history, summarize what changed from that. "
                "Otherwise say that no version history was found."
            )
        )

        messages = _grounded_summary_messages(
            user_content=(
                f"Create a short summary for artifact {artifact_id}.\n\n"
                "Return concise markdown with exactly these sections:\n"
                "## Key points\n"
                "## Decisions\n"
                "## What changed\n\n"
                "Rules:\n"
                "- Keep it short.\n"
                "- Prefer bullets.\n"
                "- Mention uncertainty if the source does not contain enough info.\n"
                "- If no decisions are present, say so.\n"
                "- If no version history is present, say so under What changed.\n\n"
                f"Current artifact excerpts / notes:\n{current_notes}"
                f"{previous_section}"
            )
        )

        return self._llm.generate(messages).strip()


def _grounded_summary_messages(user_content: str) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "You are SprintStart's grounded summarization assistant. "
                "Use only the provided source excerpts. "
                "Do not use external knowledge. "
                "Do not invent facts, decisions, changes, or links. "
                "If the source excerpts do not contain enough information, say so."
            ),
        ),
        Message(role="user", content=user_content),
    ]


def _chunk_batches(chunks: list[Chunk], max_chars: int) -> list[str]:
    ordered = sorted(chunks, key=lambda chunk: chunk.position or 0)
    batches: list[str] = []
    current = ""

    for chunk in ordered:
        block = _format_chunk(chunk)

        if len(current) + len(block) > max_chars and current:
            batches.append(current)
            current = ""

        if len(block) > max_chars:
            block = block[:max_chars] + "\n[truncated]\n"

        current += block

    if current:
        batches.append(current)

    return batches


def _format_chunk(chunk: Chunk) -> str:
    position = chunk.position if chunk.position is not None else 0
    return (
        f"\n--- Source: {chunk.filename} | chunk {position} ---\n{chunk.text.strip()}\n"
    )


def _citation_for_chunks(artifact_id: str, chunks: list[Chunk]) -> SummaryCitation:
    first = chunks[0]
    source_url = next((chunk.source_url for chunk in chunks if chunk.source_url), None)

    return SummaryCitation(
        artifact_id=artifact_id,
        filename=first.filename,
        source_url=source_url or f"/api/v1/artifacts/{artifact_id}/summary",
    )

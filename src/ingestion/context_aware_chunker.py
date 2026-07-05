"""LLM-based context-aware chunking strategy for text content.

This is an opt-in, additive alternative to the paragraph-boundary
``chunk_text`` strategy in :mod:`ingestion.chunker`. Instead of splitting on
character length alone, an LLM is given the full text (split into
paragraphs) and asked to choose semantically coherent chunk boundaries and,
optionally, a short situating "context block" for chunks that would benefit from one.

Both behaviors share a single LLM call but are independently toggleable:

- ``semantic_boundaries``: let the LLM choose paragraph groupings instead of
  the character-length accumulation used by ``chunk_text``.
- ``contextualize``: let the LLM flag chunks that need a short context block
  and prepend it to their content.

If either the text is too large for the LLM, or the LLM output cannot be
parsed, the whole call falls back to :func:`ingestion.chunker.chunk_text`.

The LLM is asked to reason over small integer *indices*, not character
offsets — LLMs are unreliable at counting characters, but reliably reproduce
small integers that are put in front of each segment in the prompt.

Note on ``semantic_boundaries=False``: the caller has already grouped the
text into its final chunks (e.g. via ``chunk_text``'s paragraph-accumulation
logic) *before* calling this module, and passes those finished chunks in as
``segments``. This way the numbering the LLM sees lines up exactly with the
chunk order index that ``context_blocks`` must key off — there is no
separate "boundaries" step to reconcile. When ``semantic_boundaries=True``,
``segments`` are the raw paragraphs instead, and the LLM's own ``boundaries``
determine the resulting chunk grouping (and thus what order index each
chunk has).
"""

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from llm.base import LLMClient, Message
from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)


class ContextAwareChunkingError(Exception):
    """Raised when the LLM output cannot be parsed/validated.

    Callers should catch this and fall back to :func:`chunker.chunk_text`.
    """


class _Payload(BaseModel):
    # Segment indices where a new chunk starts (only meaningful when the LLM
    # was asked to choose boundaries itself). Must be sorted, start at 0, and
    # contain no duplicates or out-of-range indices.
    boundaries: list[int] = Field(default_factory=list[int])
    # Chunk *order* index (0-based, as string) -> short situating sentence.
    # Chunks missing from this dict need no context block.
    context_blocks: dict[str, str] = Field(default_factory=dict)


def _build_prompt(
    segments: list[str],
    semantic_boundaries: bool,
    contextualize: bool,
) -> list[Message]:
    """Build the chat prompt for the context-aware chunking LLM call.

    Args:
        segments (list[str]):
            The text to analyze, in document order. When
            ``semantic_boundaries`` is ``True`` these are raw paragraphs
            that the LLM groups into chunks itself via ``boundaries``. When
            ``semantic_boundaries`` is ``False`` these are already the
            caller's finished chunks — the LLM only fills
            ``context_blocks`` for them, keyed by their position here.

        semantic_boundaries (bool):
            Whether the LLM should choose chunk boundaries itself. When
            ``False``, the caller has already fixed the boundaries and the
            LLM is only asked to fill ``context_blocks`` for those chunks.

        contextualize (bool):
            Whether the LLM should propose context blocks at all. When
            ``False``, the LLM is instructed to always leave
            ``context_blocks`` empty.

    Returns:
        list[Message]:
            System + user messages ready to pass to ``LLMClient.generate``.
    """
    unit_name = "paragraph" if semantic_boundaries else "chunk"
    numbered_segments = "\n\n".join(
        f"[{index}] {segment}" for index, segment in enumerate(segments)
    )

    instructions: list[str] = []
    if semantic_boundaries:
        instructions.append(
            "1. 'boundaries': a sorted list of paragraph indices where a new "
            "chunk should start, based on topic shifts, section boundaries, "
            "or other semantic breaks. It must start with 0 and contain no "
            "duplicates or indices outside the given paragraph range."
        )
    else:
        instructions.append(
            "1. 'boundaries': always return an empty list — chunk "
            "boundaries are fixed by the caller and are not part of this "
            "task."
        )

    if contextualize:
        chunk_reference = (
            "the resulting chunk's order index (0-based, counted over the "
            "chunks formed by your own 'boundaries' — NOT the paragraph "
            "index)"
            if semantic_boundaries
            else "the chunk's index as given above"
        )
        instructions.append(
            "2. 'context_blocks': a mapping from " + chunk_reference + " (as "
            "a string) to a short (one to two sentence) situating sentence "
            "that gives the chunk context it would otherwise lack (e.g. "
            "what document/section it is part of, what it is about). Only "
            "include chunks that genuinely benefit from this — "
            "self-contained chunks should be omitted from the mapping "
            "entirely."
        )
    else:
        instructions.append(
            "2. 'context_blocks': always return an empty object — "
            "contextualization is not part of this task."
        )

    system = (
        f"You analyze a document that has been split into numbered "
        f"{unit_name}s, in order to prepare it for chunking in a retrieval "
        "system.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences) with "
        "exactly two keys:\n" + "\n".join(instructions) + "\n\n"
        'JSON schema: {"boundaries": [int], "context_blocks": '
        '{"<chunk_order_index>": str}}'
    )
    user = f"Numbered {unit_name}s:\n\n{numbered_segments}"

    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _request_chunking_plan(
    segments: list[str],
    llm: LLMClient,
    semantic_boundaries: bool,
    contextualize: bool,
) -> _Payload:
    """Call the LLM and parse its response into a validated ``_Payload``.

    Raises:
        ContextAwareChunkingError:
            If the LLM output is not valid/parseable JSON matching the
            expected schema.
    """
    messages = _build_prompt(segments, semantic_boundaries, contextualize)
    raw = llm.generate(messages)

    try:
        payload = _Payload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise ContextAwareChunkingError(
            f"invalid context-aware chunking output: {exc}"
        ) from exc

    num_segments = len(segments)
    if any(
        index < 0 or index >= num_segments for index in payload.boundaries
    ) or payload.boundaries != sorted(set(payload.boundaries)):
        raise ContextAwareChunkingError(
            "LLM returned invalid boundaries: "
            f"{payload.boundaries!r} for {num_segments} segments"
        )
    if payload.boundaries and payload.boundaries[0] != 0:
        raise ContextAwareChunkingError(
            f"LLM boundaries must start at 0, got {payload.boundaries!r}"
        )

    return payload

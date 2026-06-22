from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    artifact_id: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Stable identifier for the source document. "
                "Re-ingesting with the same artifact_id replaces all existing chunks."
            ),
            examples=["sprint-42-retro"],
        ),
    ]
    filename: Annotated[
        str,
        Field(
            min_length=1,
            description="Original filename, used in citations.",
            examples=["retro.md"],
        ),
    ]
    content: str = Field(
        description=(
            "Document content as a string. "
            "For text-based files (.txt, .md, .json, .yaml, .toml) send the raw text. "
            "For image files (.png, .jpg, .jpeg, .gif, .webp, .bmp) send the file "
            "as a standard base64-encoded string. "
            "If a vision model is not configured, image chunks are silently skipped "
            "and chunk_count will be 0."
        )
    )

    @field_validator("filename")
    @classmethod
    def filename_has_no_path_separators(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("filename must not contain path separators")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "artifact_id": "sprint-42-retro",
                "filename": "retro.md",
                "content": "# Retro\n## What went well\nGood collaboration...",
            }
        }
    }


class IngestArtifactResponse(BaseModel):
    id: str = Field(description="Artifact identifier.")
    filename: str = Field(description="Original source filename.")
    content_type: str = Field(description="Detected or inferred content type.")
    source_type: str = Field(description="Source type, e.g. file, text, or url.")
    size_bytes: int = Field(description="Size of the ingested content in bytes.")
    chunk_count: int = Field(description="Number of chunks created for this artifact.")
    status: str = Field(
        description="Ingestion status, e.g. processing, completed, or failed."
    )
    created_at: str = Field(description="ISO timestamp when ingestion started.")
    updated_at: str = Field(description="ISO timestamp when ingestion last changed.")
    error_message: str | None = Field(
        default=None,
        description="Failure reason if ingestion failed.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "sprint-42-retro",
                "filename": "retro.md",
                "content_type": "text/markdown",
                "source_type": "file",
                "size_bytes": 1234,
                "chunk_count": 2,
                "status": "completed",
                "created_at": "2026-06-21T12:00:00+00:00",
                "updated_at": "2026-06-21T12:00:01+00:00",
                "error_message": None,
            }
        }
    }


class IngestChunkResponse(BaseModel):
    id: str = Field(description="Chunk identifier.")
    artifact_id: str = Field(description="Artifact this chunk belongs to.")
    filename: str = Field(description="Original source filename.")
    text: str = Field(description="Stored chunk text.")
    chunk_index: int = Field(description="Position of this chunk within the artifact.")
    vector_store_id: str = Field(description="Identifier used in the vector store.")
    kind: str = Field(description="Chunk kind, e.g. text, code, pdf, or image.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "chunk-1",
                "artifact_id": "sprint-42-retro",
                "filename": "retro.md",
                "text": "Good collaboration and faster CI feedback...",
                "chunk_index": 0,
                "vector_store_id": "chunk-1",
                "kind": "text",
            }
        }
    }


class IngestResponse(BaseModel):
    artifact_id: str = Field(description="Created or updated artifact identifier.")
    chunk_count: int = Field(
        description=(
            "Number of chunks stored. 0 indicates the file was recognised "
            "but produced no storable content, e.g. an image file when "
            "no vision model is configured."
        )
    )
    artifact: IngestArtifactResponse
    chunks: list[IngestChunkResponse]

    model_config = {
        "json_schema_extra": {
            "example": {
                "artifact_id": "sprint-42-retro",
                "chunk_count": 2,
                "artifact": {
                    "id": "sprint-42-retro",
                    "filename": "retro.md",
                    "content_type": "text/markdown",
                    "source_type": "file",
                    "size_bytes": 1234,
                    "chunk_count": 2,
                    "status": "completed",
                    "created_at": "2026-06-21T12:00:00+00:00",
                    "updated_at": "2026-06-21T12:00:01+00:00",
                    "error_message": None,
                },
                "chunks": [
                    {
                        "id": "chunk-1",
                        "artifact_id": "sprint-42-retro",
                        "filename": "retro.md",
                        "text": "Good collaboration and faster CI feedback...",
                        "chunk_index": 0,
                        "vector_store_id": "chunk-1",
                        "kind": "text",
                    }
                ],
            }
        }
    }


class HistoryEntry(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Who produced this message.")
    content: str = Field(description="Text content of the message.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "role": "user",
                "content": "What were the main blockers in sprint 42?",
            }
        }
    }


class ChatRequest(BaseModel):
    prompt: str = Field(examples=["What were the main blockers in sprint 42?"])
    context: Annotated[
        list[HistoryEntry],
        Field(
            description=(
                "Ordered conversation history for multi-turn context. "
                "Entries are chronological (oldest first) and should alternate "
                "between 'user' and 'assistant' roles. "
                "May be omitted or empty for single-turn requests."
            ),
        ),
    ] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Can you summarize that?",
                "context": [
                    {
                        "role": "user",
                        "content": "What were the main blockers in sprint 42?",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "The main blockers were missing designs "
                            "and a flaky CI pipeline."
                        ),
                    },
                ],
            }
        }
    }


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    detail: str | None = None


class TitleRequest(BaseModel):
    """
    Request model for generating a title from a user prompt.

    Args:
        prompt: Input prompt used for title generation.
        max_length: Maximum allowed length of the generated title.

    Raises:
        ValueError: If prompt is empty or contains only whitespace.
    """

    prompt: Annotated[
        str,
        Field(
            min_length=1,
            description="The input prompt used to generate the title",
            examples=["What are the main differences between REST and GraphQL?"],
        ),
    ]
    max_length: Annotated[
        int,
        Field(ge=1, le=200, description="Maximum title length"),
    ] = 60

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "What are the main differences between REST and GraphQL?",
                "max_length": 60,
            }
        }
    }

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        """
        Validate that the prompt is not blank.

        Args:
            value: Prompt string to validate.

        Raises:
            ValueError: If the prompt is empty or only whitespace.

        Returns:
            str: Validated prompt string.
        """
        if not value.strip():
            raise ValueError("prompt cannot be blank")
        return value


class TitleResponse(BaseModel):
    """
    Response model containing the generated title.

    Args:
        title: Generated title based on the provided prompt.
    """

    title: str = Field(description="From the prompt generated title.")

    model_config = {
        "json_schema_extra": {"example": {"title": "REST vs GraphQL: key differences"}}
    }


class ValidationErrorResponse(BaseModel):
    detail: str


class TokenEvent(BaseModel):
    type: Literal["token"]
    content: str = Field(
        description="A single token or short string fragment of the answer."
    )


class CitationEvent(BaseModel):
    type: Literal["citation"]
    chunk_id: str
    filename: str
    section_path: Annotated[
        str | None,
        Field(
            description='Heading breadcrumb, e.g. "Retro > Blockers". '
            "Null if not available."
        ),
    ] = None


class ToolUseEvent(BaseModel):
    type: Literal["tool_use"]
    name: Annotated[
        str,
        Field(
            description="Name of the invoked capability.",
            examples=["retrieve"],
        ),
    ]
    kind: Annotated[
        Literal["agent", "tool"],
        Field(
            description=(
                "Whether the invoked capability is a leaf 'tool' or a sub-'agent'."
            ),
        ),
    ]


class DoneEvent(BaseModel):
    type: Literal["done"]


class ErrorEvent(BaseModel):
    type: Literal["error"]
    message: str

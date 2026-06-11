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


class IngestResponse(BaseModel):
    artifact_id: str
    chunk_count: int = Field(
        description=(
            "Number of chunks stored. "
            "0 indicates the file was recognised but produced no storable content "
            "(e.g. an image file when no vision model is configured)."
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "artifact_id": "sprint-42-retro",
                "chunk_count": 4,
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
    top_k: Annotated[
        int, Field(ge=1, le=20, description="Maximum number of chunks to retrieve.")
    ] = 5
    min_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum cosine similarity score for a chunk to be included.",
        ),
    ] = 0.3
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
                "top_k": 5,
                "min_score": 0.7,
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

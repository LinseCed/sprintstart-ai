from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    artifact_id: str = Field(
        description=(
            "Stable identifier for the source document. "
            "Re-ingesting with the same artifact_id replaces all existing chunks."
        ),
        examples=["sprint-42-retro"],
    )
    filename: str = Field(
        description="Original filename, used in citations.", examples=["retro.md"]
    )
    content: str = Field(
        description="Full plain-text or Markdown content of the document."
    )

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
    chunk_count: int = Field(description="Number of chunks stored.")

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
    question: str = Field(examples=["What were the main blockers in sprint 42?"])
    top_k: int = Field(
        default=5, ge=1, le=20, description="Maximum number of chunks to retrieve."
    )
    min_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for a chunk to be included.",
    )
    history: list[HistoryEntry] = Field(
        default_factory=list,
        description=(
            "Ordered conversation history for multi-turn context. "
            "Entries are chronological (oldest first) and should alternate "
            "between 'user' and 'assistant' roles. "
            "May be omitted or empty for single-turn requests."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Can you summarize that?",
                "top_k": 5,
                "min_score": 0.7,
                "history": [
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
    section_path: str | None = Field(
        default=None,
        description=(
            'Heading breadcrumb, e.g. "Retro > Blockers". '
            "Null if not available."
        ),
    )


class DoneEvent(BaseModel):
    type: Literal["done"]


class ErrorEvent(BaseModel):
    type: Literal["error"]
    message: str

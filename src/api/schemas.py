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


class IngestResponse(BaseModel):
    artifact_id: str
    chunk_count: int = Field(description="Number of chunks stored.")


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
        )
    )


class DoneEvent(BaseModel):
    type: Literal["done"]


class ErrorEvent(BaseModel):
    type: Literal["error"]
    message: str

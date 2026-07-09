from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

if TYPE_CHECKING:
    from onboarding.models import Blueprint, PersonProfile


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
    source_role: Literal["primary", "test"] | None = Field(
        default=None,
        description=(
            "Role of this document in the corpus. 'test' marks test code and "
            "test fixtures/sample data — still searchable, but excluded from "
            "onboarding grounding. Defaults to auto-detection from the filename."
        ),
        examples=["primary"],
    )
    semantic_boundaries: bool = Field(
        default=False,
        description=(
            "Only affects text and PDF content. When true, an LLM chooses "
            "chunk boundaries based on semantic coherence (topic shifts, "
            "section boundaries) instead of the default character-length "
            "accumulation. Falls back to the default chunker if the "
            "content is too large for the LLM or the LLM output is "
            "invalid. Independently toggleable from 'contextualize'."
        ),
    )
    contextualize: bool = Field(
        default=False,
        description=(
            "Only affects text and PDF content. When true, an LLM flags "
            "which chunks would benefit from a short situating context "
            "block (Anthropic-style Contextual Retrieval) and prepends it "
            "to their content; self-contained chunks are left untouched. "
            "Falls back to the default chunker if the content is too "
            "large for the LLM or the LLM output is invalid. "
            "Independently toggleable from 'semantic_boundaries'."
        ),
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
        if not value.strip():
            raise ValueError("prompt cannot be blank")
        return value


class TitleResponse(BaseModel):
    title: str = Field(description="Generated title based on the provided prompt.")

    model_config = {
        "json_schema_extra": {"example": {"title": "REST vs GraphQL: key differences"}}
    }


class ValidationErrorResponse(BaseModel):
    detail: str


class VectorDbChunkResponse(BaseModel):
    id: str = Field(description="Chunk identifier.")
    artifact_id: str = Field(description="Artifact/document this chunk belongs to.")
    filename: str = Field(description="Original source filename.")
    text: str = Field(description="Stored chunk text.")
    position: int | None = Field(
        default=None,
        description="Optional chunk position within the source artifact.",
    )
    kind: str = Field(description="Chunk kind, e.g. text, code, pdf, or image.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "chunk-1",
                "artifact_id": "artifact-123",
                "filename": "notes.md",
                "text": "Stored chunk text...",
                "position": 0,
                "kind": "text",
            }
        }
    }


class VectorDbScoredChunkResponse(VectorDbChunkResponse):
    score: float = Field(description="Similarity score returned by vector search.")


class VectorDbChunkListResponse(BaseModel):
    items: list[VectorDbChunkResponse]
    limit: int
    offset: int
    total: int


class VectorDbStatusResponse(BaseModel):
    backend: str = Field(description="Configured vector store backend.")
    chunk_count: int = Field(description="Number of chunks currently stored.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "backend": "chroma",
                "chunk_count": 128,
            }
        }
    }


class VectorDbSearchRequest(BaseModel):
    query: Annotated[
        str,
        Field(
            min_length=1,
            description="Query text to embed and search in the vector database.",
            examples=["Where is OLLAMA_EMBED_MODEL configured?"],
        ),
    ]
    top_k: Annotated[
        int,
        Field(ge=1, le=50, description="Maximum number of chunks to return."),
    ] = 5
    min_score: Annotated[
        float,
        Field(ge=0.0, description="Minimum similarity score to include."),
    ] = 0.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Where is OLLAMA_EMBED_MODEL configured?",
                "top_k": 5,
                "min_score": 0.0,
            }
        }
    }


class VectorDbSearchResponse(BaseModel):
    items: list[VectorDbScoredChunkResponse]


class TokenEvent(BaseModel):
    type: Literal["token"]
    content: str = Field(
        description="A single token or short string fragment of the answer."
    )


class CitationEvent(BaseModel):
    type: Literal["citation"]
    chunk_id: str
    filename: str


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


class BlueprintStepSchema(BaseModel):
    id: str
    title: str
    description: str = ""
    requirement: str = "recommended"
    audience: list[str] = Field(default_factory=list)
    min_experience: str | None = None
    tags: list[str] = Field(default_factory=list)
    invariant: bool = False


class BlueprintProvenanceSchema(BaseModel):
    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list)


class BlueprintSchema(BaseModel):
    scope: str
    version: str = "0"
    source: str = "authored"
    steps: list[BlueprintStepSchema] = []
    # Carried so the backend can round-trip it: ``corpus_fingerprint`` is what
    # lets a re-generation against an unchanged corpus short-circuit.
    provenance: BlueprintProvenanceSchema | None = None

    def to_model(self) -> "Blueprint":
        """Convert the wire schema into the internal Blueprint model."""
        from onboarding.models import Blueprint, BlueprintProvenance, BlueprintStep

        return Blueprint(
            scope=self.scope,
            version=self.version,
            source=self.source,  # type: ignore[arg-type]
            steps=[
                BlueprintStep(
                    id=s.id,
                    title=s.title,
                    description=s.description,
                    requirement=s.requirement,  # type: ignore[arg-type]
                    audience=s.audience,
                    min_experience=s.min_experience,
                    tags=s.tags,
                    invariant=s.invariant,
                )
                for s in self.steps
            ],
            provenance=(
                BlueprintProvenance(**self.provenance.model_dump())
                if self.provenance is not None
                else None
            ),
        )


class SkillAssessmentSchema(BaseModel):
    name: Annotated[
        str,
        Field(min_length=1, description="Skill tag, e.g. kotlin.", examples=["kotlin"]),
    ]
    level: Annotated[
        str,
        Field(
            default="beginner",
            description=(
                "Proficiency level: beginner, intermediate, advanced, expert. "
                "Case-insensitive; unknown values are handled gracefully."
            ),
            examples=["advanced"],
        ),
    ] = "beginner"


class OnboardingPathRequest(BaseModel):
    working_area: Annotated[
        str,
        Field(
            min_length=1,
            description="The person's working area, e.g. backend, frontend, devops.",
            examples=["backend"],
        ),
    ]
    skills: list[SkillAssessmentSchema] = Field(
        default_factory=list[SkillAssessmentSchema],
        description=(
            "Optional leveled skills ({name, level}); the backend supplies the "
            "user's skill assessments so proficiency drives personalization."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional free-form tags used for step targeting.",
    )
    blueprints: list[BlueprintSchema] = Field(
        description=(
            "Active blueprints provided by the backend. The AI service is "
            "stateless — the backend owns blueprint persistence and must supply "
            "these on every request."
        ),
    )

    def to_profile(self) -> "PersonProfile":
        from onboarding.models import PersonProfile, SkillAssessment

        return PersonProfile(
            working_area=self.working_area,
            skills=[SkillAssessment(name=s.name, level=s.level) for s in self.skills],
            tags=self.tags,
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "working_area": "backend",
                "skills": [{"name": "kotlin", "level": "advanced"}],
                "tags": [],
                "blueprints": [
                    {
                        "scope": "global",
                        "version": "3",
                        "source": "generated",
                        "steps": [
                            {
                                "id": "step-abc123",
                                "title": "Set up development environment",
                                "description": "Install prerequisites",
                                "requirement": "required",
                            }
                        ],
                    }
                ],
            }
        }
    }


class GenerateBlueprintsRequest(BaseModel):
    scopes: list[str] | None = Field(
        default=None,
        description=(
            "Scopes to (re)generate, e.g. ['global', 'area:backend', 'area:frontend']. "
            "Omit to refresh 'global' plus any active blueprint scopes."
        ),
    )
    active: list[BlueprintSchema] = Field(
        default=[],
        description=(
            "The backend's currently-active blueprints. The AI service is "
            "stateless, so these drive idempotency and version numbering — pass "
            "them on every request."
        ),
    )


class StageEvent(BaseModel):
    type: Literal["stage"]
    name: Annotated[
        str,
        Field(
            description="The pipeline stage that just started.",
            examples=["retrieve"],
        ),
    ]


class PathEvent(BaseModel):
    type: Literal["path"]
    path: dict[str, object] = Field(
        description="The structured onboarding path (OnboardingPath model)."
    )
    path_yaml: str = Field(description="The onboarding path serialized to YAML.")
    quality: dict[str, object] = Field(
        description="The deterministic quality report for the path."
    )


# ── GitHub run batch ingest ───────────────────────────────────────────────────


class ArtifactRunIngestRequest(BaseModel):
    """One artifact from a completed GitHub ingestion run."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    artifact_id: str
    source_system: str
    source_id: str
    source_url: str | None = None
    artifact_type: str
    title: str | None = None
    body_text: str | None = None
    mime: str | None = None
    language: str | None = None


class RunArtifactsSyncRequest(BaseModel):
    """Batch payload sent by the backend after a GitHub ingestion run completes."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    artifacts_to_ingest: list[ArtifactRunIngestRequest]
    artifacts_to_deindex: list[str]


class ArtifactRunIngestResponse(BaseModel):
    artifact_id: str
    chunk_count: int
    status: Literal["completed", "failed"] = "completed"


class RunArtifactsSyncResponse(BaseModel):
    artifacts: list[ArtifactRunIngestResponse]


class ArtifactSummaryRequest(BaseModel):
    previous_artifact_id: str | None = Field(
        default=None,
        alias="previousArtifactId",
        description="Optional previous artifact id for change summaries.",
    )
    max_chunks: int = Field(
        default=500,
        ge=1,
        le=2000,
        alias="maxChunks",
        description="Maximum number of chunks to use for summary generation.",
    )

    model_config = ConfigDict(populate_by_name=True)


class ArtifactSummaryCitation(BaseModel):
    artifact_id: str
    filename: str
    source_url: str | None = None


class ArtifactSummaryResponse(BaseModel):
    artifact_id: str
    summary: str
    citations: list[ArtifactSummaryCitation]

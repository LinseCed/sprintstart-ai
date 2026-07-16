"""Domain models for AI-proposed competency graph elements.

Competencies and their prerequisite edges are the backend's durable graph (see
the backend's ``Competency``/``CompetencyEdge`` entities). This service never
persists graph state itself -- it proposes candidate nodes/edges for the
backend to store and a PM to review, the same proposal-only relationship
:class:`onboarding.models.Blueprint` has with the backend.

Unlike a :class:`~onboarding.models.Blueprint`, the backend's competency graph
today can only grow (no replace-whole-graph, no removal/modification) -- so
this module has no analogue of ``generation._enforce_invariants``. There is
nothing to protect a proposal run from silently dropping, because a proposal
run never removes anything; it only ever adds new candidates alongside
whatever already exists.
"""

from typing import Literal

from pydantic import BaseModel, Field

from onboarding.models import CitationRef

CompetencyKind = Literal["SKILL", "CONCEPT"]
EdgeKind = Literal["PREREQUISITE"]

ProposalStatus = Literal["proposed", "unchanged", "skipped"]


class ProposedCompetency(BaseModel):
    """A candidate competency node grounded in retrieved evidence."""

    key: str
    label: str
    description: str = ""
    kind: CompetencyKind
    repo_ref: str | None = None
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class ProposedEdge(BaseModel):
    """A candidate prerequisite edge, with a rationale for PM review."""

    from_key: str
    to_key: str
    kind: EdgeKind = "PREREQUISITE"
    rationale: str = ""


class GraphProvenance(BaseModel):
    """Why a proposal run looks the way it does; mirrors ``BlueprintProvenance``."""

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class ActiveCompetency(BaseModel):
    """A competency already in the backend's live graph.

    Drives dedup (never re-proposed as new) and is a valid prerequisite edge
    endpoint. Carries no proposal-time metadata because the backend's graph
    has none yet -- competencies are just live nodes, not versioned drafts.
    """

    key: str
    label: str
    description: str = ""
    kind: CompetencyKind
    repo_ref: str | None = None


class ActiveEdge(BaseModel):
    """A prerequisite edge already in the backend's live graph."""

    from_key: str
    to_key: str
    kind: EdgeKind = "PREREQUISITE"


class GraphProposalOutcome(BaseModel):
    """Result of one competency-graph proposal run."""

    status: ProposalStatus
    competencies: list[ProposedCompetency] = Field(
        default_factory=list[ProposedCompetency]
    )
    edges: list[ProposedEdge] = Field(default_factory=list[ProposedEdge])
    provenance: GraphProvenance | None = None
    chunks_retrieved: int = 0
    notes: list[str] = Field(default_factory=list[str])

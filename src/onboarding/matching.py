"""Deterministic hire -> starter-task fit ranking.

Given a hire's freshly-built competencies (from the backend's ledger) and the
starter-work pool (:mod:`onboarding.starter_work`'s proposals, already
PM-approved by the time this runs), rank pool tasks by fit. No LLM judgment
call the way mining/verification make one -- fit is primarily competency-key
overlap between the hire and each task, since both are grounded in the same
graph's stable keys and a direct match is the strongest signal available.
Embeddings (still an LLM-client call, just not a generation one) only break
ties, so a task the mining step left untagged doesn't rank identically to
every other untagged task.
"""

from pydantic import BaseModel, Field

from llm.base import LLMClient
from onboarding.similarity import cosine_similarity, step_text
from onboarding.starter_work import ProposedStarterTask

# Weight applied to the tie-breaking embedding similarity (0..1) relative to
# key-overlap score (also 0..1) -- small enough that any nonzero key overlap
# always outranks a zero-overlap task, however similar its text.
_SIMILARITY_TIEBREAK_WEIGHT = 0.01


class HireCompetency(BaseModel):
    """One competency a hire has met, per the backend's ledger."""

    key: str
    label: str
    description: str = ""


class RankedStarterTask(BaseModel):
    """One pool task with its fit score against a hire."""

    task: ProposedStarterTask
    score: float
    matched_competency_keys: list[str] = Field(default_factory=list[str])


def match_hire_to_pool(
    llm: LLMClient,
    hire_competencies: list[HireCompetency],
    pool: list[ProposedStarterTask],
) -> list[RankedStarterTask]:
    """Rank starter-work pool tasks by fit against a hire's competencies.

    Ties (including 0/0 overlap, e.g. an untagged task) are broken by
    embedding cosine similarity between the hire's competency labels and the
    task's title/summary, so untagged tasks don't all rank identically -- one
    topically close to what the hire knows still surfaces above one that
    isn't.
    """
    if not pool:
        return []

    hire_keys = {c.key for c in hire_competencies}
    hire_text = "; ".join(step_text(c.label, c.description) for c in hire_competencies)
    hire_embedding = llm.embed(hire_text) if hire_text.strip() else None

    task_embeddings: list[list[float]] | None = None
    if hire_embedding is not None:
        task_embeddings = llm.embed_batch(
            [step_text(task.title, task.summary) for task in pool]
        )

    ranked: list[RankedStarterTask] = []
    for i, task in enumerate(pool):
        task_keys = set(task.competency_keys)
        overlap = hire_keys & task_keys
        union = hire_keys | task_keys
        key_score = len(overlap) / len(union) if union else 0.0

        similarity = 0.0
        if hire_embedding is not None and task_embeddings is not None:
            similarity = cosine_similarity(hire_embedding, task_embeddings[i])

        score = key_score + _SIMILARITY_TIEBREAK_WEIGHT * similarity
        ranked.append(
            RankedStarterTask(
                task=task, score=score, matched_competency_keys=sorted(overlap)
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked

"""Structural documentation-coverage gap detection per component.

Called by the backend's Knowledge-Gaps insight refresh (pull-based, issue #137).
This service is stateless and sources everything from its ingestion index, so
the request carries no body.

"Insufficient" is scoped here as *structural coverage*: for each component known
to the index we determine which expected documentation categories (readme,
setup, adr, …) are present versus missing. Detection is hybrid — the LLM
classifies a component's documents into categories, with a filename heuristic as
a fallback when the LLM output can't be used.

Owners and related-question counts are deliberately NOT produced here: the
ingestion index holds no user/ownership data and this service retains no
question history. The backend enriches the returned ``component`` with those.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from store.base import VectorStore

logger = logging.getLogger(__name__)

Severity = Literal["high", "medium", "low"]

# Documentation categories every component should ideally have, ordered from
# most to least foundational. This is the "expected-type checklist" the corpus
# is measured against; ``missingTypes = expected - present``.
EXPECTED_TYPES: tuple[str, ...] = (
    "readme",
    "setup",
    "architecture",
    "adr",
    "api",
    "runbook",
)

# Categories whose absence is especially damaging for onboarding/operations and
# therefore weighs heavier in severity scoring.
CRITICAL_TYPES: frozenset[str] = frozenset({"readme", "setup"})

# A component whose newest artifact is older than this is considered stale, which
# bumps its gap severity by one notch.
_STALE_AFTER_DAYS = 180

# Bound on how many documents (and how much text per document) we feed the
# classifier, to keep the per-component prompt within a reasonable token budget.
_MAX_DOCS_PER_COMPONENT = 25
_SNIPPET_CHARS = 200

# Filename substrings mapped to categories, used as the fallback classifier.
_HEURISTIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "readme": ("readme",),
    "setup": (
        "setup",
        "install",
        "getting-started",
        "getting_started",
        "quickstart",
        "contributing",
    ),
    "architecture": ("architecture", "design"),
    "adr": ("adr", "decision-record", "decision_record"),
    "api": ("api", "openapi", "swagger", "reference"),
    "runbook": ("runbook", "playbook", "operations", "ops", "oncall", "on-call"),
}


@dataclass(frozen=True)
class KnowledgeGap:
    component: str
    missing_types: list[str]
    present_types: list[str]
    last_updated: str
    severity: Severity


def _component_of(record: ArtifactRecord) -> str | None:
    """Derive an ``owner/repo`` component from an artifact's ``source_id``.

    Source ids from the GitHub ingestion run have the shape
    ``"github:owner/repo:TYPE:..."``; the second colon-separated segment is the
    repository. Artifacts without such a segment (e.g. ad-hoc uploads) have no
    derivable component and are skipped by the caller.
    """
    source_id = record.source_id
    if not source_id:
        return None
    parts = source_id.split(":")
    if len(parts) >= 2 and "/" in parts[1]:
        return parts[1]
    return None


def _doc_snippets(
    records: list[ArtifactRecord],
    store: VectorStore,
) -> list[tuple[str, str]]:
    """Return ``(filename, text snippet)`` pairs for a component's documents."""
    snippets: list[tuple[str, str]] = []
    for record in records[:_MAX_DOCS_PER_COMPONENT]:
        text = ""
        chunks = store.list_chunks_by_artifact(record.id, limit=1)
        if chunks:
            text = chunks[0].text[:_SNIPPET_CHARS].replace("\n", " ").strip()
        snippets.append((record.filename, text))
    return snippets


def _heuristic_present(records: list[ArtifactRecord]) -> set[str]:
    """Filename-based fallback classification of present categories."""
    present: set[str] = set()
    for record in records:
        name = record.filename.lower()
        for category, keywords in _HEURISTIC_KEYWORDS.items():
            if any(keyword in name for keyword in keywords):
                present.add(category)
    return present


def _build_classify_prompt(
    component: str,
    snippets: list[tuple[str, str]],
) -> list[Message]:
    docs = "\n".join(
        f"- {filename}: {text}" if text else f"- {filename}"
        for filename, text in snippets
    )
    categories = ", ".join(EXPECTED_TYPES)
    system = (
        "You assess the documentation coverage of a software component. You are "
        "given the component's documents (filename and an optional snippet). "
        "Decide which of these documentation categories the component already "
        f"has substantive coverage for: {categories}.\n\n"
        "A category counts as present only if at least one document genuinely "
        "serves that purpose — do not guess from a filename alone if the snippet "
        "contradicts it. Return STRICT JSON only (no prose, no markdown fences) "
        'with this schema: {"present": [<category>, ...]}. Use only categories '
        "from the list above."
    )
    user = f"Component: {component}\n\nDocuments:\n{docs}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _classify_present(
    component: str,
    records: list[ArtifactRecord],
    llm: LLMClient,
    store: VectorStore,
) -> set[str]:
    """Classify which expected categories the component covers.

    Uses the LLM as the primary classifier and falls back to the filename
    heuristic when the LLM output can't be parsed. ``LLMUnavailableError`` is
    allowed to propagate so the endpoint can surface a 503.
    """
    snippets = _doc_snippets(records, store)
    raw = llm.generate(_build_classify_prompt(component, snippets))
    try:
        payload = json.loads(extract_json_object(raw))
        if not isinstance(payload, dict):
            raise ValueError("classification output is not an object")
        present = cast(dict[str, object], payload).get("present")
        if not isinstance(present, list):
            raise ValueError("'present' is not a list")
        return {str(item) for item in cast(list[object], present)} & set(
            EXPECTED_TYPES
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "Knowledge-gap classification for %s fell back to heuristic: %s",
            component,
            exc,
        )
        return _heuristic_present(records)


def _is_stale(last_updated: str) -> bool:
    try:
        updated = datetime.fromisoformat(last_updated)
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    age_days = (datetime.now(UTC) - updated).days
    return age_days > _STALE_AFTER_DAYS


def _severity(missing: list[str], last_updated: str) -> Severity:
    """Rank a gap.

    Score accrues from the number of missing categories, with an extra penalty
    when a critical category (readme/setup) is missing and when the component's
    newest document is stale.
    """
    missing_set = set(missing)
    score = len(missing_set)
    if missing_set & CRITICAL_TYPES:
        score += 2
    if _is_stale(last_updated):
        score += 1
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def detect_knowledge_gaps(
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
) -> list[KnowledgeGap]:
    """Detect per-component documentation-coverage gaps across the index."""
    components: dict[str, list[ArtifactRecord]] = {}
    for record in metadata_store.list_artifacts():
        component = _component_of(record)
        if component is None:
            continue
        components.setdefault(component, []).append(record)

    gaps: list[KnowledgeGap] = []
    for component, records in sorted(components.items()):
        present = _classify_present(component, records, llm, store)
        missing = [t for t in EXPECTED_TYPES if t not in present]
        if not missing:
            continue
        last_updated = max(record.updated_at for record in records)
        gaps.append(
            KnowledgeGap(
                component=component,
                missing_types=missing,
                present_types=[t for t in EXPECTED_TYPES if t in present],
                last_updated=last_updated,
                severity=_severity(missing, last_updated),
            )
        )

    gaps.sort(key=lambda g: (_SEVERITY_RANK[g.severity], g.component))
    return gaps

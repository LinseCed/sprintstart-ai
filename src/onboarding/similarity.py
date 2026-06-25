"""Reusable similarity utilities for cross-scope step deduplication.

Two modes:

* **Embedding cosine similarity** — used at generation time where an LLM client
  is available. Higher precision, higher cost.
* **Token-set overlap (Jaccard)** — used at pipeline/serve time as a fast,
  deterministic safety net. No external calls.
"""

import math
import re

# Deterministic threshold for the pipeline's token-overlap gate.  Two steps
# whose Jaccard index exceeds this are considered semantic duplicates.
OVERLAP_THRESHOLD: float = 0.55

# Generation-time embedding threshold (stricter to avoid false positives).
SIMILARITY_THRESHOLD: float = 0.75

# Minimal stopword set — common English function words only.  Domain terms
# like "install", "setup", "verify" are intentionally excluded because they
# carry semantic weight for distinguishing onboarding steps.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "in",
        "is",
        "it",
        "not",
        "of",
        "on",
        "or",
        "set",
        "that",
        "the",
        "this",
        "to",
        "up",
        "was",
        "with",
        "you",
        "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def step_text(title: str, description: str) -> str:
    """Combine title and description into a single comparable string."""
    combined = f"{title}. {description}".strip()
    return combined.rstrip(".")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


def text_overlap(a: str, b: str) -> float:
    """Jaccard index over normalised, stopword-filtered token sets."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

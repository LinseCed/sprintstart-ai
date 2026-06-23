"""Shared helpers for parsing LLM text output."""


def extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from an LLM response.

    Raises ``ValueError`` when no ``{...}`` block is found.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in LLM output")
    return text[start : end + 1]

"""Recover tool calls a model emitted as text instead of as structured calls.

Some models served through OpenAI-/Ollama-compatible endpoints (notably DeepSeek
via OpenRouter) emit tool calls in their own XML-ish markup *inside the message
content* rather than in the API's structured ``tool_calls`` field. The endpoint
then fails to lift them out, so the raw markup leaks into the assistant's visible
answer, e.g.::

    <｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="search_docs">
    <｜｜DSML｜｜parameter name="query" string="true">...</｜｜DSML｜｜parameter>
    </｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>

The buddy agent drives itself off structured ``tool_calls``; when they arrive as
text like this the tool is never run and the markup is shown to the hire. This
module detects that markup, parses it back into :class:`ToolCall` objects, and
strips it from the text, so the agent loop can run the tool as if it had been
returned structurally.

The parser keys on the structural tokens (``invoke name=``, ``parameter name=``),
never on the surrounding delimiters, so it tolerates the various pipe/prefix
wrappers different models dress them in.
"""

import json
import re
from uuid import uuid4

from llm.base import ToolCall

# One ``invoke name="…"`` block and everything up to its matching close (or the
# end of the string, if the model never closed it).
_INVOKE_RE = re.compile(
    r'invoke\s+name="([^"]+)"(.*?)(?:</[^>]*\binvoke\b[^>]*>|$)',
    re.IGNORECASE | re.DOTALL,
)
# ``parameter name="…"…>value</…parameter>`` inside one invoke block.
_PARAM_RE = re.compile(
    r'parameter\s+name="([^"]+)"[^>]*>(.*?)</[^>]*\bparameter\b[^>]*>',
    re.IGNORECASE | re.DOTALL,
)
# The visible start of the leaked block, so the prose before it (if any) is kept.
_BLOCK_START_RE = re.compile(
    r'<[^>]*\b(?:tool_calls|invoke)\b|invoke\s+name="',
    re.IGNORECASE,
)
# Corroborating markup that tells a genuine leaked call apart from prose that
# merely happens to contain the substring ``invoke name="…"``.
_MARKUP_RE = re.compile(
    r"tool_calls|DSML|</[^>]*\binvoke\b|parameter\s+name=",
    re.IGNORECASE,
)


def _coerce(value: str) -> object:
    """Parse a parameter value as JSON when it looks like one, else keep the text."""
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped


def recover_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Extract tool calls a model wrote into ``content`` as markup.

    Returns the recovered calls and the text with the markup removed. When no
    leaked call is present, returns ``([], content)`` unchanged — callers should
    only fall back to this when the structured ``tool_calls`` field was empty.
    """
    if not content or not _MARKUP_RE.search(content):
        return [], content

    invokes = list(_INVOKE_RE.finditer(content))
    if not invokes:
        return [], content

    calls = [
        ToolCall(
            id=f"call_{uuid4().hex}",
            name=match.group(1).strip(),
            arguments={
                name: _coerce(value)
                for name, value in _PARAM_RE.findall(match.group(2))
            },
        )
        for match in invokes
    ]

    cut = invokes[0].start()
    start = _BLOCK_START_RE.search(content)
    if start is not None and start.start() < cut:
        cut = start.start()
    return calls, content[:cut].rstrip()


__all__ = ["recover_tool_calls"]

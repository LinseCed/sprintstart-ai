"""Server-Sent Events framing shared by the streaming orchestrators."""

import json


def sse_event(payload: dict[str, object]) -> str:
    """Frame a JSON payload as a single SSE ``data:`` event."""
    return f"data: {json.dumps(payload)}\n\n"

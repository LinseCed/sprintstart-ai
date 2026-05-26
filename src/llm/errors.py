class LLMUnavailableError(Exception):
    def __init__(self, host: str | None, cause: Exception | None = None) -> None:
        self.host = host
        detail = f" (caused by: {cause})" if cause else ""
        super().__init__(f"LLM backend unreachable at {host!r}{detail}")

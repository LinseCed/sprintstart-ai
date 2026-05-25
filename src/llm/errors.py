class LLMUnavailableError(Exception):
    def __init__(self, host:str, cause: Exception | None = None) -> None:
        self.host = host
        self.cause = cause
        detail = f" (caused by: {cause})" if cause else ""
        super().__init__(f"LLM backend unreachable at {host!r}{detail}")

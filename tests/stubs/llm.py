class StubLLMClient:
    def __init__(
        self,
        generate_response: str = "stub answer",
        embedding: list[float] | None = None,
    ) -> None:
        self.generate_response = generate_response
        self.embedding = embedding or [0.0] * 768

    def generate(self, prompt: str) -> str:
        return self.generate_response

    def embed(self, text: str) -> list[float]:
        return self.embedding
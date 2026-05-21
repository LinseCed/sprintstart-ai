class StubLLMClient:
    def __init__(self, generate_response: str = "stub answer") -> None:
        self.generate_response = generate_response

    def generate(self, prompt: str) -> str:
        return self.generate_response

    def embed(self, text: str) -> list[float]:
        return [0.0] * 768

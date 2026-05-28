from llm.base import Message
from tests.stubs.llm import StubLLMClient


def test_generate_returns_configured_response():
    client = StubLLMClient(generate_response="hello world")
    messages = [Message(role="user", content="any prompt")]
    assert client.generate(messages) == "hello world"


def test_embed_returns_correct_dimension():
    client = StubLLMClient()
    result = client.embed("any text")
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)

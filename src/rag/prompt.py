from llm.base import Message
from rag.types import ScoredChunk


def build_messages(
    question: str,
    chunks: list[ScoredChunk],
    history: list[Message],
) -> list[Message]:
    messages: list[Message] = []

    if chunks:
        context = "\n\n".join(chunk.text for chunk in chunks)
        messages.append(
            Message(
                role="system",
                content=f"Answer based on the following context:\n\n{context}",
            )
        )

    messages.extend(history)
    messages.append(Message(role="user", content=question))

    return messages

import base64
from collections.abc import Iterator
from typing import Any

from openai import OpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError


def _to_openai_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    openai_messages: list[ChatCompletionMessageParam] = []

    for message in messages:
        openai_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": message["content"],
        }
        openai_messages.append(openai_message)

    return openai_messages


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str,
        embed_model: str,
        vision_model: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.vision_model = vision_model

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )

    def generate(self, messages: list[Message]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=_to_openai_messages(messages),
            )

            content = response.choices[0].message.content
            return content or ""

        except OpenAIError as exc:
            raise LLMUnavailableError(
                f"OpenAI-compatible chat backend unavailable: {exc}"
            ) from exc

    def stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            stream: Iterator[ChatCompletionChunk] = self.client.chat.completions.create(
                model=self.chat_model,
                messages=_to_openai_messages(messages),
                stream=True,
            )

            for event in stream:
                if not event.choices:
                    continue

                delta: ChoiceDelta = event.choices[0].delta
                content = delta.content

                if content:
                    yield content

        except OpenAIError as exc:
            raise LLMUnavailableError(
                f"OpenAI-compatible streaming backend unavailable: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=text,
            )

            return list(response.data[0].embedding)

        except OpenAIError as exc:
            raise LLMUnavailableError(
                f"OpenAI-compatible embedding backend unavailable: {exc}"
            ) from exc

    def caption_image(self, image_bytes: bytes) -> str:
        if self.vision_model is None:
            raise LLMUnavailableError(
                "OpenAI-compatible vision model is not configured"
            )

        try:
            encoded = base64.b64encode(image_bytes).decode("ascii")

            messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded}",
                            },
                        },
                    ],
                }
            ]

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
            )

            content = response.choices[0].message.content
            return content or ""

        except OpenAIError as exc:
            raise LLMUnavailableError(
                f"OpenAI-compatible vision backend unavailable: {exc}"
            ) from exc
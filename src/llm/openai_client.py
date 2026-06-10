import base64
import imghdr
from collections.abc import Iterator
from typing import Any

from openai import OpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")

    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"

    return normalized


def _to_openai_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    openai_messages: list[ChatCompletionMessageParam] = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            system_message: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": content,
            }
            openai_messages.append(system_message)
        elif role == "assistant":
            assistant_message: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": content,
            }
            openai_messages.append(assistant_message)
        else:
            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": content,
            }
            openai_messages.append(user_message)

    return openai_messages


def _detect_image_mime_type(image_bytes: bytes) -> str:
    image_type = imghdr.what(None, image_bytes)

    if image_type is None:
        raise LLMUnavailableError("Could not detect image MIME type")

    if image_type == "jpg":
        image_type = "jpeg"

    return f"image/{image_type}"


class OpenAIClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str,
        embed_model: str,
        vision_model: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.vision_model = vision_model

        self.client = OpenAI(
            base_url=self.base_url,
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
                "OpenAI-compatible backend unavailable during chat "
                f"using model {self.chat_model!r} at {self.base_url}: {exc}"
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
                "OpenAI-compatible backend unavailable during streaming "
                f"using model {self.chat_model!r} at {self.base_url}: {exc}"
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
                "OpenAI-compatible backend unavailable during embedding "
                f"using model {self.embed_model!r} at {self.base_url}: {exc}"
            ) from exc

    def caption_image(self, image_bytes: bytes) -> str:
        if self.vision_model is None:
            raise LLMUnavailableError(
                "OpenAI-compatible vision model is not configured"
            )

        try:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            mime_type = _detect_image_mime_type(image_bytes)

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
                                "url": f"data:{mime_type};base64,{encoded}",
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
                "OpenAI-compatible backend unavailable during vision "
                f"using model {self.vision_model!r} at {self.base_url}: {exc}"
            ) from exc

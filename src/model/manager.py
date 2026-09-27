"""Chat model abstraction, normalized payloads, and factory.

This module defines the :class:`ChatModel` protocol used throughout the
application. Concrete provider integrations live under
:mod:`src.model.providers` and are registered through
:mod:`src.model.registry`. Business code must depend only on this module and
the registered provider identifiers; it must never reference a concrete SDK.
"""

from __future__ import annotations

import typing

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Sequence

    from langchain_core.messages import BaseMessage

    from src.settings import ModelChatSettings

from src.model.registry import resolve_factory

ProviderFactory = typing.Callable[["ModelChatSettings"], "ChatModel"]


class ChatRequest(BaseModel):
    """Provider-independent chat completion request."""

    messages: list[dict[str, typing.Any]] = Field(default_factory=list)
    tools: list[dict[str, typing.Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    extra: dict[str, typing.Any] = Field(default_factory=dict)

    def to_langchain_messages(self) -> list[BaseMessage]:
        """Convert stored message dictionaries into LangChain message objects."""

        output: list[BaseMessage] = []
        for message in self.messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                output.append(SystemMessage(content=content))
            elif role == "user":
                output.append(HumanMessage(content=content))
            elif role == "assistant":
                kwargs: dict[str, typing.Any] = {"content": content}
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    kwargs["tool_calls"] = tool_calls
                output.append(AIMessage(**kwargs))
            elif role == "tool":
                output.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=message.get("tool_call_id", ""),
                    )
                )
            else:
                output.append(HumanMessage(content=content))
        return output


class ChatResponse(BaseModel):
    """Normalized chat model response."""

    content: str = ""
    tool_calls: list[dict[str, typing.Any]] = Field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    finish_reason: str | None = None
    provider: str | None = None
    raw: dict[str, typing.Any] = Field(default_factory=dict)

    def has_tool_calls(self) -> bool:
        """Return True when the model requested at least one tool invocation."""

        return bool(self.tool_calls)


class ChatModel:
    """Chat model protocol. Concrete providers must implement these methods."""

    name: str = "base"

    def invoke(self, request: ChatRequest) -> ChatResponse:
        """Execute a synchronous chat completion."""

        raise NotImplementedError

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        """Execute an asynchronous chat completion."""

        raise NotImplementedError

    def bind_tools(self, tools: Sequence[dict[str, typing.Any]]) -> ChatModel:
        """Return a model configured to use the supplied tools."""

        raise NotImplementedError


def create_chat_model(settings: ModelChatSettings) -> ChatModel:
    """Build a chat model from validated settings."""

    factory = resolve_factory(settings.provider)
    return factory(settings)


__all__ = [
    "ChatModel",
    "ChatRequest",
    "ChatResponse",
    "ProviderFactory",
    "create_chat_model",
]

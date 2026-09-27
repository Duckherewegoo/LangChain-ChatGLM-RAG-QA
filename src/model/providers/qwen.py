"""Alibaba Tongyi Qwen provider.

Qwen can be reached through DashScope's OpenAI-compatible endpoint or its
native SDK. This module provides the compatible-mode adapter; the native SDK
implementation can be added later without changing consumers.
"""

from __future__ import annotations

import typing

from src.model.manager import ChatModel
from src.model.providers.openai_compatible import OpenAICompatibleChatModel
from src.model.registry import register_provider

if typing.TYPE_CHECKING:
    from src.settings import ModelChatSettings

_QWEN_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenCompatibleChatModel(OpenAICompatibleChatModel):
    """Qwen model adapter using DashScope's OpenAI-compatible endpoint."""

    name = "qwen"

    def __init__(self, settings: ModelChatSettings) -> None:
        normalized = settings
        if not settings.base_url:
            normalized = settings.model_copy(update={"base_url": _QWEN_COMPATIBLE_BASE_URL})
        super().__init__(normalized)


register_provider("qwen", QwenCompatibleChatModel)


class QwenNativeChatModel(ChatModel):
    """Placeholder for a future native DashScope SDK adapter.

    This class exists to make the provider slot explicit. Implement the native
    client only after confirming the required SDK and authentication method.
    """

    name = "qwen_native"

    def __init__(self, settings: ModelChatSettings) -> None:
        raise NotImplementedError(
            "Native Qwen/DashScope adapter is not implemented. "
            "Use provider 'qwen' for the OpenAI-compatible endpoint."
        )

    def invoke(self, request: typing.Any) -> typing.Any:
        raise NotImplementedError

    async def ainvoke(self, request: typing.Any) -> typing.Any:
        raise NotImplementedError

    def bind_tools(self, tools: typing.Any) -> ChatModel:
        raise NotImplementedError


__all__ = ["QwenCompatibleChatModel", "QwenNativeChatModel"]

"""DeepSeek provider adapter.

DeepSeek exposes an OpenAI-compatible API, so this provider mainly codifies
its official endpoint and recommended model identifiers. Behavior-specific
tuning can be added here without affecting the base protocol adapter.
"""

from __future__ import annotations

import typing

from src.model.providers.openai_compatible import OpenAICompatibleChatModel
from src.model.registry import register_provider

if typing.TYPE_CHECKING:
    from src.settings import ModelChatSettings

_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekChatModel(OpenAICompatibleChatModel):
    """DeepSeek model adapter using its OpenAI-compatible endpoint."""

    name = "deepseek"

    def __init__(self, settings: ModelChatSettings) -> None:
        normalized = settings
        if not settings.base_url:
            normalized = settings.model_copy(update={"base_url": _DEEPSEEK_DEFAULT_BASE_URL})
        super().__init__(normalized)


register_provider("deepseek", DeepSeekChatModel)


__all__ = ["DeepSeekChatModel"]

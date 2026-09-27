"""ZhipuAI ChatGLM provider.

This module registers a provider that points at the ZhipuAI OpenAI-compatible
endpoint by default. It inherits common protocol handling from
:mod:`src.model.providers.openai_compatible` and overrides only the defaults
that distinguish ChatGLM.
"""

from __future__ import annotations

import typing

from src.model.errors import ModelConfigurationError
from src.model.providers.openai_compatible import OpenAICompatibleChatModel
from src.model.registry import register_provider

if typing.TYPE_CHECKING:
    from src.settings import ModelChatSettings

_ZHIPU_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/compatibles/v1"


class ZhipuChatModel(OpenAICompatibleChatModel):
    """ChatGLM-compatible model using ZhipuAI's OpenAI-compatible endpoint."""

    name = "zhipu"

    def __init__(self, settings: ModelChatSettings) -> None:
        normalized = self._normalize_settings(settings)
        super().__init__(normalized)

    @staticmethod
    def _normalize_settings(settings: ModelChatSettings) -> ModelChatSettings:
        if not settings.is_configured:
            raise ModelConfigurationError(
                "Zhipu API key is empty; set MODEL_CHAT_API_KEY or configure model_chat.api_key.",
                details={"provider": settings.provider, "model_name": settings.model_name},
            )
        if settings.base_url:
            return settings
        return settings.model_copy(update={"base_url": _ZHIPU_DEFAULT_BASE_URL})


register_provider("zhipu", ZhipuChatModel)


__all__ = ["ZhipuChatModel"]

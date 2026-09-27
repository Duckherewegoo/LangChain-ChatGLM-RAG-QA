"""Non-OpenAI protocol provider placeholders.

Doubao/Volcano Engine, Tencent Hunyuan, and Yuanbao may require native SDKs or
private protocols. These classes define explicit extension points. Implement
them only after confirming the current SDK, endpoint, authentication method,
and tool-call response contract. Registering a placeholder before it is ready
prevents consumers from silently assuming the capability exists.
"""

from __future__ import annotations

import typing

from src.model.errors import ModelConfigurationError
from src.model.manager import ChatModel
from src.model.registry import register_provider

if typing.TYPE_CHECKING:
    from src.settings import ModelChatSettings

_PLACEHOLDER_MESSAGE = (
    "The {provider} provider is not implemented. "
    "Add a protocol-specific adapter before selecting it in configuration."
)


class _ProtocolPlaceholder(ChatModel):
    """Common behavior for unimplemented protocol adapters."""

    provider_name: str = "placeholder"

    def __init__(self, settings: ModelChatSettings) -> None:
        raise ModelConfigurationError(
            _PLACEHOLDER_MESSAGE.format(provider=self.provider_name),
            details={"provider": settings.provider, "model_name": settings.model_name},
        )

    def invoke(self, request: typing.Any) -> typing.Any:
        raise NotImplementedError

    async def ainvoke(self, request: typing.Any) -> typing.Any:
        raise NotImplementedError

    def bind_tools(self, tools: typing.Any) -> ChatModel:
        raise NotImplementedError


class DoubaoChatModel(_ProtocolPlaceholder):
    """Volcano Engine Doubao provider placeholder."""

    name = "doubao"
    provider_name = "doubao"


class HunyuanChatModel(_ProtocolPlaceholder):
    """Tencent Hunyuan provider placeholder."""

    name = "hunyuan"
    provider_name = "hunyuan"


class YuanbaoChatModel(_ProtocolPlaceholder):
    """Tencent Yuanbao provider placeholder."""

    name = "yuanbao"
    provider_name = "yuanbao"


register_provider("doubao", DoubaoChatModel)
register_provider("hunyuan", HunyuanChatModel)
register_provider("yuanbao", YuanbaoChatModel)


__all__ = ["DoubaoChatModel", "HunyuanChatModel", "YuanbaoChatModel"]

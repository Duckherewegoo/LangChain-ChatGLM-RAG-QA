"""非 OpenAI 协议厂商占位符。

豆包（火山引擎）、腾讯混元、腾讯元宝可能需要原生 SDK 或私有协议。
这些类定义了显式的扩展点。只有在确认当前的 SDK、端点、鉴权方式以及
工具调用的响应契约后，才应实现它们。在准备就绪前注册占位符，
可以防止调用方静默地假设该能力已经存在。
"""

from __future__ import annotations

import typing

from model.errors import ModelConfigurationError
from model.protocol import ChatModel, ChatRequest, ChatResponse
from model.registry import register_provider

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from collections.abc import Sequence

    from settings import ModelChatSettings

_PLACEHOLDER_MESSAGE = (
    "提供者 {provider} 尚未实现。"
    "请在配置中选择该提供者之前，添加对应协议的具体适配器。"
)


class _ProtocolPlaceholder:
    """未实现的协议适配器通用行为。"""

    name: str = "placeholder"
    provider_name: str = "placeholder"
    model_name: str = "placeholder"

    def __init__(self, settings: ModelChatSettings) -> None:
        # Fail Fast：一旦被实例化立刻抛出配置错误，阻止后续的无效调用
        raise ModelConfigurationError(
            _PLACEHOLDER_MESSAGE.format(provider=self.provider_name),
            details={"provider": settings.provider, "model_name": settings.model_name},
        )

    def invoke(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    def bind_tools(self, tools: Sequence[dict[str, typing.Any]]) -> ChatModel:
        raise NotImplementedError


class DoubaoChatModel(_ProtocolPlaceholder):
    """火山引擎豆包提供者占位符。"""

    name = "doubao"
    provider_name = "doubao"
    model_name = "doubao-placeholder"


class HunyuanChatModel(_ProtocolPlaceholder):
    """腾讯混元提供者占位符。"""

    name = "hunyuan"
    provider_name = "hunyuan"
    model_name = "hunyuan-placeholder"


class YuanbaoChatModel(_ProtocolPlaceholder):
    """腾讯元宝提供者占位符。"""

    name = "yuanbao"
    provider_name = "yuanbao"
    model_name = "yuanbao-placeholder"


# 导入即注册：占位符也会进入注册表，但在实例化时会显式报错
register_provider("doubao", DoubaoChatModel)
register_provider("hunyuan", HunyuanChatModel)
register_provider("yuanbao", YuanbaoChatModel)


__all__ = [
    "DoubaoChatModel",
    "HunyuanChatModel",
    "YuanbaoChatModel",
]

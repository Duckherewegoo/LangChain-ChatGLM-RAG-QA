"""阿里通义千问（Qwen）厂商适配器。

Qwen 可以通过 DashScope 的 OpenAI 兼容端点或其原生 SDK 访问。
本模块提供兼容模式（Compatible Mode）的适配器；
原生 SDK 的实现可以在未来添加，而无需更改调用方。
"""

from __future__ import annotations

import typing

from model.errors import ModelConfigurationError
from model.protocol import ChatModel, ChatRequest, ChatResponse
from model.providers.openai_compatible import OpenAICompatibleChatModel
from model.registry import register_provider

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from collections.abc import Sequence

    from settings import ModelChatSettings

_QWEN_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_QWEN_DEFAULT_MODEL = "qwen-turbo"


class QwenCompatibleChatModel(OpenAICompatibleChatModel):
    """使用 DashScope OpenAI 兼容端点的 Qwen 模型适配器。"""

    name = "qwen"

    def __init__(self, settings: "ModelChatSettings") -> None:
        # 1. 防御性校验：Fail Fast
        if not settings.api_key:
            raise ModelConfigurationError(
                "Qwen 提供者缺少必要的 API Key 配置。",
                details={"provider": self.name}
            )

        # 2. 补充默认配置
        updates = {}
        if not settings.base_url:
            updates["base_url"] = _QWEN_COMPATIBLE_BASE_URL
        if not settings.model_name:
            updates["model_name"] = _QWEN_DEFAULT_MODEL

        # 3. 安全覆盖
        normalized = settings.model_copy(update=updates) if updates else settings
        super().__init__(normalized)


# 导入即注册：由 router 层动态触发懒加载
register_provider("qwen", QwenCompatibleChatModel)


class QwenNativeChatModel:
    """未来原生 DashScope SDK 适配器的占位符。

    此类的存在是为了让该提供者的扩展槽位显式化。
    只有在确认所需的 SDK 和鉴权方式后，才应实现原生客户端。
    """

    name = "qwen_native"
    model_name = "qwen-native-placeholder"

    def __init__(self, settings: "ModelChatSettings") -> None:
        # 统一抛出配置错误，以便上层 health 检查能正确归类
        raise ModelConfigurationError(
            "原生 Qwen/DashScope 适配器尚未实现。请使用 'qwen' 提供者以调用 OpenAI 兼容端点。",
            details={"provider": settings.provider, "model_name": settings.model_name},
        )

    def invoke(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    def bind_tools(self, tools: Sequence[dict[str, typing.Any]]) -> ChatModel:
        raise NotImplementedError


__all__ = ["QwenCompatibleChatModel", "QwenNativeChatModel"]

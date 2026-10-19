"""DeepSeek 厂商适配器。

DeepSeek 暴露了与 OpenAI 完全兼容的 API 接口，
因此本适配器主要固化其官方端点与推荐的模型标识符。
特定行为的调优可以在此处添加，而不影响基础的协议适配器。
"""

from __future__ import annotations

import typing

from src.exceptions import ModelConfigurationError
from src.model.providers.openai_compatible import OpenAICompatibleChatModel
from src.model.registry import register_provider

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from src.settings import ModelChatSettings

_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


class DeepSeekChatModel(OpenAICompatibleChatModel):
    """使用 OpenAI 兼容端点的 DeepSeek 模型适配器。"""

    name = "deepseek"

    def __init__(self, settings: "ModelChatSettings") -> None:
        # 1. 防御性校验：Fail Fast，避免发起无效请求才报错
        if not settings.api_key:
            raise ModelConfigurationError(
                "DeepSeek 提供者缺少必要的 API Key 配置。",
                details={"provider": self.name}
            )

        # 2. 补充默认配置
        updates = {}
        if not settings.base_url:
            updates["base_url"] = _DEEPSEEK_DEFAULT_BASE_URL
        if not settings.model_name:
            updates["model_name"] = _DEEPSEEK_DEFAULT_MODEL

        # 3. 安全覆盖，不影响外部传入的原始 settings 对象
        normalized = settings.model_copy(update=updates) if updates else settings

        super().__init__(normalized)


# 导入即注册：由 router 层动态触发懒加载
register_provider("deepseek", DeepSeekChatModel)


__all__ = ["DeepSeekChatModel"]

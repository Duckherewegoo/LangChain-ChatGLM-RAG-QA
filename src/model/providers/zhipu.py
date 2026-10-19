"""智谱 AI ChatGLM 厂商适配器。

本模块默认注册一个指向智谱 AI OpenAI 兼容端点的提供者。
它继承自 src.model.providers.openai_compatible 的通用协议处理逻辑，
仅覆盖用于区分 ChatGLM 的默认配置。
"""

from __future__ import annotations

import typing

from src.model.errors import ModelConfigurationError
from src.model.providers.openai_compatible import OpenAICompatibleChatModel
from src.model.registry import register_provider

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from src.settings import ModelChatSettings

# 注意：请核对智谱官方最新文档，确认该兼容模式端点是否准确
_ZHIPU_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/compatibles/v1"
_ZHIPU_DEFAULT_MODEL = "glm-4"


class ZhipuChatModel(OpenAICompatibleChatModel):
    """使用智谱 AI OpenAI 兼容端点的 ChatGLM 模型适配器。"""

    name = "zhipu"

    def __init__(self, settings: "ModelChatSettings") -> None:
        normalized = self._normalize_settings(settings)
        super().__init__(normalized)

    @staticmethod
    def _normalize_settings(settings: "ModelChatSettings") -> "ModelChatSettings":
        """校验必要配置并填充默认值。"""
        # 1. 防御性校验：Fail Fast
        if not settings.is_configured:
            raise ModelConfigurationError(
                "智谱 AI 的 API Key 为空；请设置 MODEL_CHAT_API_KEY 或配置 model_chat.api_key。",
                details={"provider": settings.provider, "model_name": settings.model_name},
            )

        updates = {}

        # 2. 补充默认 Base URL
        if not settings.base_url:
            updates["base_url"] = _ZHIPU_DEFAULT_BASE_URL

        # 3. 补充默认模型名
        if not settings.model_name:
            updates["model_name"] = _ZHIPU_DEFAULT_MODEL

        # 4. 安全覆盖：仅在确实有字段需要更新时才执行模型复制
        if updates:
            return settings.model_copy(update=updates)
        return settings


# 导入即注册：由 router 层动态触发懒加载
register_provider("zhipu", ZhipuChatModel)


__all__ = ["ZhipuChatModel"]

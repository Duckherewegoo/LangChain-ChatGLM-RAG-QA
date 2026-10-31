"""智谱 AI ChatGLM 厂商适配器。

本模块默认注册一个指向智谱 AI OpenAI 兼容端点的提供者。
它继承自 model.providers.openai_compatible 的通用协议处理逻辑，
仅覆盖用于区分 ChatGLM 的默认配置。
"""

from __future__ import annotations

import typing

from model.providers.openai_compatible import OpenAICompatibleChatModel
from model.registry import register_provider

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from settings import ModelChatSettings

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
        """填充智谱默认端点和模型名；必要字段校验交给基类的 _assert_configured。"""
        updates: dict[str, typing.Any] = {}
        if not settings.model_name:
            updates["model_name"] = _ZHIPU_DEFAULT_MODEL
        if not settings.base_url:
            updates["base_url"] = _ZHIPU_DEFAULT_BASE_URL
        return settings.model_copy(update=updates) if updates else settings


# 导入即注册：由 router 层动态触发懒加载
register_provider("zhipu", ZhipuChatModel)


__all__ = ["ZhipuChatModel"]

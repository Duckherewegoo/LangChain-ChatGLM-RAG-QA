"""Chat model factory and provider registry resolution.

负责工厂模式的创建与具体实现类的注册解析。
"""

from __future__ import annotations

import typing

from src.model.protocol import ChatModel
from src.model.registry import resolve_factory

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from src.settings import ModelChatSettings

# 工厂函数的类型别名
ProviderFactory = typing.Callable[["ModelChatSettings"], ChatModel]


def create_chat_model(settings: "ModelChatSettings") -> ChatModel:
    """
    根据已验证的配置构建聊天模型。

    该函数会查找对应 provider 的注册工厂，并实例化相应的模型适配器。
    """
    factory = resolve_factory(settings.provider)
    return factory(settings)


__all__ = [
    "ProviderFactory",
    "create_chat_model",
]

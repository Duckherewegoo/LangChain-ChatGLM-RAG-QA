"""模型提供者注册表与发现机制。

该模块是映射“提供者标识符”到“模型工厂”的唯一权威入口。
它刻意不包含任何具体的 SDK 导入，以此保持厂商依赖的隔离，
并允许在缺失可选依赖时显式抛出注册失败。
"""

from __future__ import annotations

import threading
import typing

from src.exceptions import ConfigurationError

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from src.model.protocol import ChatModel
    from src.settings import ModelChatSettings

    ProviderFactory = typing.Callable[[ModelChatSettings], ChatModel]

# 线程锁保护全局注册表
_registry_lock = threading.RLock()
_REGISTRY: dict[str, ProviderFactory] = {}
_ALIASES: dict[str, str] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """将工厂函数注册到指定的提供者标识符下。"""
    if not name or not isinstance(name, str):
        raise TypeError("提供者名称必须是非空字符串。")
    if not callable(factory):
        raise TypeError("提供者工厂必须是可调用对象。")

    with _registry_lock:
        _REGISTRY[name.strip().lower()] = factory


def register_alias(alias: str, canonical_name: str) -> None:
    """将备用别名映射到已注册的提供者名称。"""
    key = alias.strip().lower()
    target = canonical_name.strip().lower()

    with _registry_lock:
        if target not in _REGISTRY:
            raise ConfigurationError(
                f"无法将别名 {alias!r} 映射到未注册的提供者 {canonical_name!r}。",
                details={"alias": alias, "canonical_name": canonical_name},
            )
        _ALIASES[key] = target


def resolve_factory(provider: str) -> ProviderFactory:
    """返回指定提供者的工厂函数，支持解析别名。"""
    if not isinstance(provider, str) or not provider.strip():
        raise ConfigurationError(
            "模型提供者必须是非空字符串。",
            details={"provider": provider},
        )

    key = provider.strip().lower()

    with _registry_lock:
        canonical = _ALIASES.get(key, key)
        factory = _REGISTRY.get(canonical)

        if factory is None:
            raise ConfigurationError(
                f"不支持的模型提供者: {provider!r}。",
                details={
                    "available_providers": sorted(_REGISTRY),
                    "aliases": dict(_ALIASES),
                },
            )
        return factory


def available_providers() -> list[str]:
    """返回已注册的提供者标识符列表。"""
    with _registry_lock:
        return sorted(_REGISTRY)


def available_aliases() -> dict[str, str]:
    """返回当前的提供者别名映射。"""
    with _registry_lock:
        return dict(_ALIASES)


def is_registered(name: str) -> bool:
    """如果提供者名称已注册，则返回 True。"""
    with _registry_lock:
        return name.strip().lower() in _REGISTRY


def clear_registry() -> None:
    """清空所有提供者和别名。
    
    仅供测试使用。生产代码绝对不应调用此函数。
    """
    with _registry_lock:
        _REGISTRY.clear()
        _ALIASES.clear()


__all__ = [
    "available_aliases",
    "available_providers",
    "clear_registry",
    "is_registered",
    "register_alias",
    "register_provider",
    "resolve_factory",
]

"""多模型配置与路由。

本模块负责将业务用例解析为配置的模型、构建主备执行顺序，
并决定模型是否可用于 Agent 的工具循环。
它不负责执行请求，也不直接调用厂商 SDK。

纯数据类型（如 ModelRoutingEntry、UseCaseBinding 等）位于 src.model.routing_types，
以便 src.settings 可以在不引入整个模型层的情况下声明 model_router 字段。
"""

from __future__ import annotations

import importlib
import threading
import typing

from src.exceptions import ConfigurationError, ResourceNotFoundError
from src.model.routing_types import (
    ModelCapabilitySettings,
    ModelRouterSettings,
    ModelRoutingEntry,
    UseCaseBinding,
)

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from src.model.fallback import FallbackExecutor
    from src.model.protocol import ChatModel


class ModelRoute:
    """针对单个用例解析后的执行计划。"""

    def __init__(
        self,
        use_case: str,
        primary: "ChatModel",
        primary_name: str,
        secondaries: list[tuple[str, "ChatModel"]],
    ) -> None:
        self.use_case = use_case
        self.primary = primary
        self.primary_name = primary_name
        self.secondaries = secondaries

    def ordered_names(self) -> list[str]:
        """按执行顺序返回模型名称。"""
        return [self.primary_name, *(name for name, _ in self.secondaries)]

    def build_executor(self, *, max_attempts_per_provider: int = 1) -> "FallbackExecutor":
        """使用解析后的模型链创建 Fallback 执行器。"""
        from src.model.fallback import FallbackExecutor

        return FallbackExecutor(
            self.primary,
            [model for _, model in self.secondaries],
            max_attempts_per_provider=max_attempts_per_provider,
        )


class ModelRouter:
    """解析用例到模型链，不负责实际调用。"""

    def __init__(self, configuration: ModelRouterSettings) -> None:
        self._configuration = configuration
        self._cache: dict[str, "ChatModel"] = {}
        # 线程锁保护模型缓存，防止并发实例化
        self._cache_lock = threading.Lock()

    @property
    def configuration(self) -> ModelRouterSettings:
        return self._configuration

    def get_model(self, name: str) -> "ChatModel":
        """按配置的名称返回缓存的模型实例。"""
        # 快速路径：无锁读取缓存
        if name in self._cache:
            return self._cache[name]

        with self._cache_lock:
            # 双重检查锁：防止在等待锁的过程中其他线程已经实例化完毕
            if name in self._cache:
                return self._cache[name]

            entry = self._configuration.models.get(name)
            if entry is None:
                raise ResourceNotFoundError(
                    f"模型 {name!r} 尚未配置。",
                    details={"available_models": sorted(self._configuration.models)},
                )

            _ensure_providers_registered()

            # 延迟导入工厂函数和协议，避免顶层循环依赖
            from src.model.manager import create_chat_model

            # 复用 routing_types 中的转换方法，避免重复实现
            model = create_chat_model(entry.to_provider_settings())
            self._cache[name] = model
            return model

    def route(self, use_case: str | None = None) -> ModelRoute:
        """为业务用例解析模型链。"""
        target_use_case = use_case or self._configuration.default_use_case
        binding = self._configuration.use_cases.get(target_use_case)

        if binding is None:
            raise ConfigurationError(
                f"用例 {target_use_case!r} 没有配置模型绑定。",
                details={"available_use_cases": sorted(self._configuration.use_cases)},
            )

        primary = self.get_model(binding.primary)
        secondaries: list[tuple[str, "ChatModel"]] = []
        for fallback_name in binding.fallbacks:
            secondaries.append((fallback_name, self.get_model(fallback_name)))

        return ModelRoute(binding.use_case, primary, binding.primary, secondaries)

    def supports_tools(self, name: str) -> bool:
        """当指定的模型支持 Function Calling 时返回 True。"""
        entry = self._configuration.models.get(name)
        if entry is None:
            return False
        return entry.capabilities.supports_tools

    def has_use_case(self, use_case: str) -> bool:
        """当用例已被配置时返回 True。"""
        return use_case in self._configuration.use_cases

    def clear_cache(self) -> None:
        """清空已缓存的模型实例。"""
        with self._cache_lock:
            self._cache.clear()


def _ensure_providers_registered() -> None:
    """导入 provider 包以触发注册副作用。

    providers 子包在被导入时会执行厂商适配器的注册逻辑。
    采用懒加载导入可以避免在进程启动时强行要求所有可选 SDK 必须存在。
    """
    try:
        importlib.import_module("src.model.providers")
    except ImportError as exc:
        raise ConfigurationError(
            "无法加载模型提供者（Provider）包，请检查依赖是否已正确安装。",
            details={"error": str(exc)},
        ) from exc


def select_fallback_chain(
    preferred: str,
    available: typing.Iterable[str],
    *,
    exclude: typing.Container[str] = (),
) -> list[str]:
    """调整模型名称的顺序，将首选模型置于最前。

    这是一个纯排序辅助函数，用于当健康报告需要覆盖静态配置顺序时。
    它不会实例化任何模型。
    """
    names = [name for name in available if name not in exclude]
    ordered: list[str] = []
    if preferred in names:
        ordered.append(preferred)
    ordered.extend(name for name in names if name != preferred)
    return ordered


__all__ = [
    "ModelCapabilitySettings",
    "ModelRoute",
    "ModelRouter",
    "ModelRouterSettings",
    "ModelRoutingEntry",
    "UseCaseBinding",
    "select_fallback_chain",
]

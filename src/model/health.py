"""模型健康检查与就绪决策。

健康检查负责验证配置，并执行一次轻量级的模型往返通信，
不执行检索，也不生成大段内容。检查是只读的：不修改被检查模型的状态。
结果按提供者分组，
并统一通过健康检查端点对外暴露。
"""

from __future__ import annotations

import time
import typing

from pydantic import BaseModel, Field

from model.errors import classify_model_error
from model.protocol import ChatModel, ChatRequest
from observability import get_logger

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

# 全局健康检查缓存：key 为 (provider, use_tools)，避免两种检查共用缓存。
_HEALTH_CACHE: dict[tuple[str, bool], tuple[float, "ProviderHealth"]] = {}
# 缓存有效期（秒），避免 K8s 探针频繁请求导致消耗真实 API 额度
_CACHE_TTL = 60.0


class ProviderHealth(BaseModel):
    """单个提供者的健康状态。

    provider 是调用方使用的配置模型键（model_router.models 的 key），
    不一定是底层 provider 名；model_name 是具体模型标识。
    """

    provider: str
    model_name: str
    configured: bool = False
    reachable: bool = False
    latency_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    supports_tools: bool = False


class HealthReport(BaseModel):
    """一个或多个提供者的聚合健康报告。"""

    providers: list[ProviderHealth] = Field(default_factory=list)
    primary: str | None = None
    available: list[str] = Field(default_factory=list)
    degraded: bool = False

    def is_healthy(self) -> bool:
        """当至少有一个提供者可达时返回 True。"""
        return bool(self.available)

    def best_available(self) -> str | None:
        """返回第一个可达的提供者标识符。"""
        return self.available[0] if self.available else None


def check_provider(
    provider: str, model: ChatModel, *, use_tools: bool = False, force_refresh: bool = False
) -> ProviderHealth:
    """对单个模型执行轻量级健康检查（带缓存机制）。"""

    now = time.perf_counter()
    cache_key = (provider, use_tools)

    cached = None if force_refresh else _HEALTH_CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    started = time.perf_counter()

    # ChatModel 实例存在即意味着构造时已通过配置校验。
    configured = True

    health = ProviderHealth(
        provider=provider,
        model_name=model.model_name,
        configured=configured,
    )

    # 构造极简的测试请求
    request = ChatRequest(
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
        tools=(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "description": "noop",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            if use_tools
            else []
        ),
    )

    try:
        # ChatRequest.tools 由 provider 层合并到 payload，无需 bind_tools，避免污染实例。
        response = model.invoke(request)
    except Exception as exc:
        classified = classify_model_error(exc, provider=provider)
        health.error_type = classified.error_type
        health.error_message = str(exc)
        logger.warning(
            f"model_health_check_failed | provider={provider} | "
            f"error_type={classified.error_type} | error={exc}"
        )
        _HEALTH_CACHE[cache_key] = (time.perf_counter(), health)
        return health

    health.reachable = True
    health.latency_seconds = time.perf_counter() - started

    # 仅当模型实际返回工具调用时才判为支持工具；ping 为 noop，模型可能不调用。
    health.supports_tools = bool(response.tool_calls)

    _HEALTH_CACHE[cache_key] = (time.perf_counter(), health)
    return health


def check_chain(
    primary_provider: str,
    primary: ChatModel,
    secondaries: Iterable[tuple[str, ChatModel]] = (),
    *,
    tool_enabled: bool = False,
    force_refresh: bool = False,
) -> HealthReport:
    """检查主提供者以及所有备用候选者。"""

    report = HealthReport(primary=primary_provider)

    # 检查主模型
    primary_health = check_provider(
        primary_provider, primary, use_tools=tool_enabled, force_refresh=force_refresh
    )
    report.providers.append(primary_health)
    if primary_health.reachable:
        report.available.append(primary_provider)

    # 检查备用模型
    for provider, model in secondaries:
        health = check_provider(
            provider, model, use_tools=tool_enabled, force_refresh=force_refresh
        )
        report.providers.append(health)
        if health.reachable:
            report.available.append(provider)

    # 如果没有任何模型可用，或者主模型挂了但备用模型活着，则处于降级状态
    report.degraded = not report.available or (
        not primary_health.reachable and bool(report.available)
    )
    return report


def select_provider(report: HealthReport, preferred: str | None) -> str | None:
    """选择提供者，仅当首选提供者可达时才优先使用。

    预留接口：当前执行链由 container 静态构造，不消费本函数的排序结果；
    若未来接入"健康检查驱动的动态路由"，由 container 调用本函数重排 provider 链。
    """
    reachable_providers = {health.provider for health in report.providers if health.reachable}

    if preferred and preferred in reachable_providers:
        return preferred
    return report.best_available()


def clear_health_cache() -> None:
    """清空健康检查缓存（仅供测试使用）。"""
    _HEALTH_CACHE.clear()


__all__ = [
    "HealthReport",
    "ProviderHealth",
    "check_chain",
    "check_provider",
    "clear_health_cache",
    "select_provider",
]

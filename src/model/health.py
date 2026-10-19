"""模型健康检查与就绪决策。

健康检查负责验证配置，并执行一次轻量级的模型往返通信，
不执行检索，也不生成大段内容。结果按提供者分组，
并统一通过健康检查端点对外暴露。
"""

from __future__ import annotations

import time
import typing

from pydantic import BaseModel, Field

from src.model.errors import classify_model_error
from src.model.protocol import ChatModel, ChatRequest, ChatResponse
from src.observability import get_logger

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

# 全局健康检查缓存：记录每个 provider 上次检查的时间和结果
# 格式: { provider_name: (timestamp, ProviderHealth) }
_HEALTH_CACHE: dict[str, tuple[float, "ProviderHealth"]] = {}
# 缓存有效期（秒），避免 K8s 探针频繁请求导致消耗真实 API 额度
_CACHE_TTL = 60.0


class ProviderHealth(BaseModel):
    """单个提供者的健康状态。"""

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

    # 检查缓存
    if not force_refresh and provider in _HEALTH_CACHE:
        cached_time, cached_health = _HEALTH_CACHE[provider]
        if now - cached_time < _CACHE_TTL:
            return cached_health

    started = time.perf_counter()

    # 优先使用协议中定义的 model_name，若没有则回退到 name
    model_name = getattr(model, "model_name", model.name)
    configured = bool(getattr(model, "is_configured", True))

    health = ProviderHealth(
        provider=provider,
        model_name=model_name,
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
        # 如果要求测试工具，则先绑定工具
        bound_model = model.bind_tools(request.tools) if use_tools else model
        response = bound_model.invoke(request)
    except Exception as exc:
        classified = classify_model_error(exc, provider=provider)
        health.error_type = classified.error_type
        health.error_message = str(exc)
        logger.warning(
            "model_health_check_failed",
            provider=provider,
            error_type=classified.error_type,
            error=str(exc),
        )
        _HEALTH_CACHE[provider] = (time.perf_counter(), health)
        return health

    health.reachable = True
    health.latency_seconds = time.perf_counter() - started

    # 只有当模型真正返回了工具调用（或者明确没有报错），才认为支持工具
    health.supports_tools = use_tools and isinstance(response, ChatResponse)

    _HEALTH_CACHE[provider] = (time.perf_counter(), health)
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
    """选择提供者，仅当首选提供者可达时才优先使用。"""
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

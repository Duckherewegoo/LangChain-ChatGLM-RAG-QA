"""Model health checks and readiness decisions.

A health check validates configuration and a lightweight model round trip
without performing retrieval or generating long content. Results are grouped by
provider and surfaced through the unified health endpoint.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field

from src.model.errors import classify_model_error
from src.model.manager import ChatModel, ChatRequest, ChatResponse
from src.observability import get_logger

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)


class ProviderHealth(BaseModel):
    """Health status for a single provider."""

    provider: str
    model_name: str
    configured: bool = False
    reachable: bool = False
    latency_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    supports_tools: bool = False


class HealthReport(BaseModel):
    """Aggregate health report for one or more providers."""

    providers: list[ProviderHealth] = Field(default_factory=list)
    primary: str | None = None
    available: list[str] = Field(default_factory=list)
    degraded: bool = False

    def is_healthy(self) -> bool:
        """Return True when at least one provider is reachable."""

        return bool(self.available)

    def best_available(self) -> str | None:
        """Return the first reachable provider identifier."""

        return self.available[0] if self.available else None


def check_provider(
    provider: str, model: ChatModel, *, use_tools: bool = False
) -> ProviderHealth:
    """Perform a lightweight health check against one model."""

    started = _monotonic()
    health = ProviderHealth(
        provider=provider,
        model_name=_safe_model_name(model),
        configured=bool(getattr(getattr(model, "_settings", None), "is_configured", True)),
    )
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
        if use_tools:
            model.bind_tools(request.tools)
        response = model.invoke(request)
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
        return health

    health.reachable = True
    health.latency_seconds = _monotonic() - started
    health.supports_tools = isinstance(response, ChatResponse)
    return health


def check_chain(
    primary_provider: str,
    primary: ChatModel,
    secondaries: Iterable[tuple[str, ChatModel]] = (),
    *,
    tool_enabled: bool = False,
) -> HealthReport:
    """Check the primary provider and all fallback candidates."""

    report = HealthReport(primary=primary_provider)
    primary_health = check_provider(primary_provider, primary, use_tools=tool_enabled)
    report.providers.append(primary_health)
    if primary_health.reachable:
        report.available.append(primary_provider)

    for provider, model in secondaries:
        health = check_provider(provider, model, use_tools=tool_enabled)
        report.providers.append(health)
        if health.reachable:
            report.available.append(provider)

    report.degraded = not report.available or (
        primary_health.reachable is False and bool(report.available)
    )
    return report


def select_provider(report: HealthReport, preferred: str | None) -> str | None:
    """Pick a provider, honoring preference only when it is reachable."""

    if preferred and preferred in {health.provider for health in report.providers if health.reachable}:
        return preferred
    return report.best_available()


def _safe_model_name(model: ChatModel) -> str:
    return getattr(getattr(model, "_settings", None), "model_name", model.name)


def _monotonic() -> float:
    import time

    return time.perf_counter()


__all__ = [
    "HealthReport",
    "ProviderHealth",
    "check_chain",
    "check_provider",
    "select_provider",
]

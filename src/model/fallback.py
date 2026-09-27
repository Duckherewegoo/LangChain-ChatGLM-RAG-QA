"""Model fallback execution.

This module implements a sequential fallback chain. It retries transient
failures against the preferred model, then escalates to secondary providers.
Permanent failures—such as authentication, content policy, and configuration
errors—are not masked by retrying or failing over, because they would recur on
every provider.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field

from src.observability import get_logger

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Iterable

    from src.model.manager import ChatModel, ChatRequest, ChatResponse


def _resolve_chat_response() -> type:
    """Resolve ChatResponse at runtime to avoid circular import with manager."""
    from src.model.manager import ChatResponse

    return ChatResponse


# Use a string forward reference; Pydantic resolves it after manager is loaded.
ChatResponseRef = "ChatResponse"

logger = get_logger(__name__)


class FallbackAttempt(BaseModel):
    """Record of one attempt within a fallback chain."""

    provider: str
    model_name: str
    status: str
    error_type: str | None = None
    error_message: str | None = None
    elapsed_seconds: float = 0.0
    response: "ChatResponse | None" = None


class FallbackResult(BaseModel):
    """Outcome of executing a fallback chain."""

    response: "ChatResponse | None" = None
    succeeded: bool = False
    attempts: list[FallbackAttempt] = Field(default_factory=list)
    final_error: str | None = None

    def summary(self) -> dict[str, typing.Any]:
        """Return a serializable summary for logging and diagnostics."""

        return {
            "succeeded": self.succeeded,
            "attempts": [attempt.model_dump() for attempt in self.attempts],
            "final_error": self.final_error,
        }


class FallbackExecutor:
    """Execute a model request across a prioritized chain."""

    def __init__(
        self,
        primary: ChatModel,
        secondaries: Iterable[ChatModel] = (),
        *,
        max_attempts_per_provider: int = 1,
    ) -> None:
        if max_attempts_per_provider < 1:
            raise ValueError("max_attempts_per_provider must be at least 1.")

        self._chain: list[ChatModel] = [primary, *secondaries]
        if not self._chain:
            raise ValueError("Fallback chain must contain at least one model.")
        self._max_attempts = max_attempts_per_provider

    @property
    def providers(self) -> list[str]:
        """Return provider identifiers in execution order."""

        return [model.name for model in self._chain]

    def invoke(self, request: ChatRequest) -> FallbackResult:
        """Execute ``request`` synchronously across the fallback chain."""

        result = FallbackResult()
        for model in self._chain:
            for attempt_index in range(self._max_attempts):
                attempt = self._try_once(model, request, attempt_index)
                result.attempts.append(attempt)
                if attempt.status == "success":
                    result.succeeded = True
                    result.response = attempt.response  # type: ignore[assignment]
                    return result
                if not self._should_retry(attempt):
                    break
        result.final_error = result.attempts[-1].error_message if result.attempts else "No model available"
        return result

    async def ainvoke(self, request: ChatRequest) -> FallbackResult:
        result = FallbackResult()
        for model in self._chain:
            for attempt_index in range(self._max_attempts):
                attempt = await self._try_once_async(model, request, attempt_index)
                result.attempts.append(attempt)
                if attempt.status == "success":
                    result.succeeded = True
                    result.response = attempt.response  # type: ignore[assignment]
                    return result
                if not self._should_retry(attempt):
                    break
        result.final_error = result.attempts[-1].error_message if result.attempts else "No model available"
        return result

    def _try_once(self, model: ChatModel, request: ChatRequest, attempt_index: int) -> FallbackAttempt:
        started = _monotonic()
        try:
            response = model.invoke(request)
        except Exception as exc:
            return self._record_failure(model, exc, started, attempt_index)
        return FallbackAttempt(
            provider=model.name,
            model_name=_safe_model_name(model),
            status="success",
            elapsed_seconds=_monotonic() - started,
            response=response,
        )

    async def _try_once_async(
        self, model: ChatModel, request: ChatRequest, attempt_index: int
    ) -> FallbackAttempt:
        started = _monotonic()
        try:
            response = await model.ainvoke(request)
        except Exception as exc:
            return self._record_failure(model, exc, started, attempt_index)
        return FallbackAttempt(
            provider=model.name,
            model_name=_safe_model_name(model),
            status="success",
            elapsed_seconds=_monotonic() - started,
            response=response,
        )

    @staticmethod
    def _record_failure(
        model: ChatModel, exc: BaseException, started: float, attempt_index: int
    ) -> FallbackAttempt:
        from src.model.errors import classify_model_error

        classified = classify_model_error(exc, provider=model.name)
        logger.warning(
            "model_fallback_attempt_failed",
            provider=model.name,
            model_name=_safe_model_name(model),
            attempt=attempt_index + 1,
            error_type=classified.error_type,
            retryable=classified.retryable,
            error=str(exc),
        )
        return FallbackAttempt(
            provider=model.name,
            model_name=_safe_model_name(model),
            status="failed",
            error_type=classified.error_type,
            error_message=str(exc),
            elapsed_seconds=_monotonic() - started,
        )

    @staticmethod
    def _should_retry(attempt: FallbackAttempt) -> bool:
        if attempt.status == "success":
            return False
        if attempt.error_type in {
            "authentication",
            "configuration",
            "content_policy",
            "payload",
        }:
            return False
        return attempt.error_type in {"timeout", "connection", "rate_limit", "unknown"}


def _safe_model_name(model: ChatModel) -> str:
    return getattr(getattr(model, "_settings", None), "model_name", model.name)


def _monotonic() -> float:
    import time

    return time.perf_counter()


__all__ = ["FallbackAttempt", "FallbackExecutor", "FallbackResult"]


# Resolve forward references now that manager has been imported.
ChatResponse = _resolve_chat_response()
FallbackAttempt.model_rebuild()
FallbackResult.model_rebuild()

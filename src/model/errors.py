"""模型层异常与错误分类。

异常被刻意设计为细粒度，使得调用方能够区分：
配置错误、客户端错误、限流、内容策略拒绝、以及不可恢复的提供者故障，
而无需解析错误消息字符串。

设计约定：
- `ClassifiedError.error_type` 是稳定类别（8 个值），供调用方分支；
- `ClassifiedError.exception_type` 是原始异常类名，供诊断；
- 审计字段（如 `FallbackAttempt.error_type`）使用 `error_type`，不使用 `error_code`。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from base.exceptions import ModelInvocationError, ModelNotConfiguredError

# ==================== 细粒度异常定义 ====================

class ModelConfigurationError(ModelNotConfiguredError):
    """缺少必要的模型配置，或配置不一致。"""
    status_code = 500
    error_code = "model_configuration_error"


class ModelAPIConnectionError(ModelInvocationError):
    """与提供者建立传输层连接失败。"""
    status_code = 502
    error_code = "model_connection_error"


class ModelTimeoutError(ModelInvocationError):
    """提供者在配置的超时时间内未响应。"""
    status_code = 504
    error_code = "model_timeout"


class ModelRateLimitError(ModelInvocationError):
    """提供者因配额或限流拒绝了请求。"""
    status_code = 429
    error_code = "model_rate_limited"


class ModelAuthenticationError(ModelInvocationError):
    """API Key 或凭证被拒绝；属于调用失败，不是未配置。"""
    status_code = 401
    error_code = "model_authentication_error"


class ModelContentPolicyError(ModelInvocationError):
    """提供者出于内容策略原因阻断了请求或响应。"""
    status_code = 400
    error_code = "model_content_policy"


class ModelPayloadError(ModelInvocationError):
    """请求或响应无法被解析。"""
    status_code = 400
    error_code = "model_payload_error"


# ==================== 分类结果模型 ====================

class ClassifiedError(BaseModel):
    """classify_model_error 的类型化结果。"""

    model_config = {"arbitrary_types_allowed": True}

    exception_type: str
    error_type: str
    message: str
    retryable: bool
    provider: str | None = None
    status_code: int = 500
    original: BaseException | None = Field(default=None, exclude=True)

    def to_exception(self) -> Exception:
        """返回合适的领域异常。"""
        mapping = {
            "configuration": ModelConfigurationError,
            "authentication": ModelAuthenticationError,
            "connection": ModelAPIConnectionError,
            "timeout": ModelTimeoutError,
            "rate_limit": ModelRateLimitError,
            "content_policy": ModelContentPolicyError,
            "payload": ModelPayloadError,
            "unknown": ModelInvocationError,
        }
        exception_type = mapping.get(self.error_type, ModelInvocationError)
        exception = exception_type(self.message)
        if self.original is not None:
            exception.__cause__ = self.original
        return exception


# ==================== 关键词表 ====================

# 已被前面显式关键词分支覆盖的词（timeout / connection / 429 等）不在此列。
# 命中后无法归入具体类别，统一按 connection 处理（可重试）。
_TRANSIENT_KEYWORDS = (
    "temporar", "try again", "503", "502", "504",
    "internal server error", "overloaded", "capacity",
    "econnreset", "socket",
)

# error_code → 稳定 error_type。新增细粒度异常时，在这里补一行映射即可。
_ERROR_CODE_TO_TYPE = {
    "model_authentication_error": "authentication",
    "model_connection_error": "connection",
    "model_timeout": "timeout",
    "model_rate_limited": "rate_limit",
    "model_content_policy": "content_policy",
    "model_payload_error": "payload",
}

_RETRYABLE_TYPES = frozenset({"timeout", "connection", "rate_limit"})


# ==================== 分类器 ====================

def classify_model_error(
    exc: BaseException, *, provider: str | None = None
) -> ClassifiedError:
    """将提供者抛出的异常映射为稳定的错误类别。"""

    message = str(exc).strip()
    lowered = message.lower()

    # 分支 1：配置类（不可重试）
    if isinstance(exc, ModelNotConfiguredError):
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="configuration",
            message=message or "模型未配置。",
            retryable=False,
            provider=provider,
            status_code=exc.status_code,
            original=exc,
        )

    # 分支 2：其他模型层异常（error_code → 稳定类型）
    if isinstance(exc, ModelInvocationError):
        category = _ERROR_CODE_TO_TYPE.get(exc.error_code, "unknown")
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type=category,
            message=message or "模型调用失败。",
            retryable=category in _RETRYABLE_TYPES,
            provider=provider,
            status_code=exc.status_code,
            original=exc,
        )

    # 分支 3：基于消息关键词的启发式分类
    if "invalid api key" in lowered or "authentication" in lowered or "unauthorized" in lowered or "401" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="authentication",
            message=message,
            retryable=False,
            provider=provider,
            status_code=401,
            original=exc,
        )
    if "rate limit" in lowered or "429" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="rate_limit",
            message=message,
            retryable=True,
            provider=provider,
            status_code=429,
            original=exc,
        )
    if "content policy" in lowered or "moderation" in lowered or ("400" in lowered and "content" in lowered):
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="content_policy",
            message=message,
            retryable=False,
            provider=provider,
            status_code=400,
            original=exc,
        )
    if "timed out" in lowered or "timeout" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="timeout",
            message=message,
            retryable=True,
            provider=provider,
            status_code=504,
            original=exc,
        )
    if "connection" in lowered or "network" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="connection",
            message=message,
            retryable=True,
            provider=provider,
            status_code=502,
            original=exc,
        )
    if any(kw in lowered for kw in _TRANSIENT_KEYWORDS):
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="connection",
            message=message,
            retryable=True,
            provider=provider,
            status_code=502,
            original=exc,
        )

    # 兜底：未知错误
    return ClassifiedError(
        exception_type=type(exc).__name__,
        error_type="unknown",
        message=message or "未知的模型提供者错误。",
        retryable=False,
        provider=provider,
        status_code=500,
        original=exc,
    )


def is_retryable(exc: BaseException, *, provider: str | None = None) -> bool:
    """当错误是瞬时故障时返回 True。"""
    return classify_model_error(exc, provider=provider).retryable


__all__ = [
    "ClassifiedError",
    "ModelAPIConnectionError",
    "ModelAuthenticationError",
    "ModelConfigurationError",
    "ModelContentPolicyError",
    "ModelPayloadError",
    "ModelRateLimitError",
    "ModelTimeoutError",
    "classify_model_error",
    "is_retryable",
]

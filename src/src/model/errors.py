"""Model-layer exceptions and error classification.

Exceptions are intentionally narrow so that callers can distinguish
configuration mistakes, client errors, rate limits, content policy rejections,
and unrecoverable provider failures without parsing messages.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.exceptions import ModelInvocationError, ModelNotConfiguredError

_RETRYABLE_KEYWORDS = (
    "timeout",
    "timed out",
    "connection",
    "network",
    "temporar",
    "try again",
    "503",
    "502",
    "504",
    "internal server error",
    "overloaded",
    "capacity",
    "econnreset",
    "socket",
)


class ModelConfigurationError(ModelNotConfiguredError):
    """Required model configuration is missing or inconsistent."""

    status_code = 500
    error_code = "model_configuration_error"


class ModelAPIConnectionError(ModelInvocationError):
    """Transport-level failure contacting the model provider."""

    status_code = 502
    error_code = "model_connection_error"


class ModelTimeoutError(ModelInvocationError):
    """The model provider did not respond within the configured timeout."""

    status_code = 504
    error_code = "model_timeout"


class ModelRateLimitError(ModelInvocationError):
    """The provider rejected the request due to quota or rate limiting."""

    status_code = 429
    error_code = "model_rate_limited"


class ModelAuthenticationError(ModelNotConfiguredError):
    """API key or credential was rejected."""

    status_code = 401
    error_code = "model_authentication_error"


class ModelContentPolicyError(ModelInvocationError):
    """The provider blocked the request or response for policy reasons."""

    status_code = 400
    error_code = "model_content_policy"


class ModelPayloadError(ModelInvocationError):
    """The request or response could not be interpreted."""

    status_code = 400
    error_code = "model_payload_error"


class ClassifiedError(BaseModel):
    """Typed outcome of :func:`classify_model_error`."""

    model_config = {"arbitrary_types_allowed": True}

    exception_type: str
    error_type: str
    message: str
    retryable: bool
    provider: str | None = None
    status_code: int = 500

    def to_exception(self) -> Exception:
        """Return an appropriate domain exception."""

        mapping = {
            "configuration": ModelConfigurationError,
            "authentication": ModelAuthenticationError,
            "connection": ModelAPIConnectionError,
            "timeout": ModelTimeoutError,
            "rate_limit": ModelRateLimitError,
            "content_policy": ModelContentPolicyError,
            "payload": ModelPayloadError,
        }
        exception_type = mapping.get(self.error_type, ModelInvocationError)
        return exception_type(self.message)


def classify_model_error(
    exc: BaseException, *, provider: str | None = None
) -> ClassifiedError:
    """Map a provider exception to a stable error category."""

    message = str(exc).strip()
    lowered = message.lower()

    if isinstance(exc, ModelNotConfiguredError):
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="configuration",
            message=message or "Model is not configured.",
            retryable=False,
            provider=provider,
            status_code=exc.status_code,
        )

    if isinstance(exc, ModelAuthenticationError):
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="authentication",
            message=message or "Model authentication failed.",
            retryable=False,
            provider=provider,
            status_code=401,
        )
    if isinstance(exc, ModelTimeoutError):
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="timeout",
            message=message or "Model request timed out.",
            retryable=True,
            provider=provider,
            status_code=504,
        )
    if isinstance(exc, ModelInvocationError):
        category = exc.error_code
        retryable = category in {"timeout", "connection", "rate_limit"}
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type=category,
            message=message or "Model invocation failed.",
            retryable=retryable,
            provider=provider,
            status_code=exc.status_code,
        )

    if "authentication" in lowered or "unauthorized" in lowered or "401" in lowered or "invalid api key" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="authentication",
            message=message,
            retryable=False,
            provider=provider,
            status_code=401,
        )
    if "rate limit" in lowered or "429" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="rate_limit",
            message=message,
            retryable=True,
            provider=provider,
            status_code=429,
        )
    if "authentication" in lowered or "unauthorized" in lowered or "401" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="authentication",
            message=message,
            retryable=False,
            provider=provider,
            status_code=401,
        )
    if "content policy" in lowered or "moderation" in lowered or ("400" in lowered and "content" in lowered):
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="content_policy",
            message=message,
            retryable=False,
            provider=provider,
            status_code=400,
        )
    if "timed out" in lowered or "timeout" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="timeout",
            message=message,
            retryable=True,
            provider=provider,
            status_code=504,
        )
    if "connection" in lowered or "network" in lowered:
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="connection",
            message=message,
            retryable=True,
            provider=provider,
            status_code=502,
        )
    if any(keyword in lowered for keyword in _RETRYABLE_KEYWORDS):
        return ClassifiedError(
            exception_type=type(exc).__name__,
            error_type="connection",
            message=message,
            retryable=True,
            provider=provider,
            status_code=502,
        )

    return ClassifiedError(
        exception_type=type(exc).__name__,
        error_type="unknown",
        message=message or "Unknown model provider error.",
        retryable=False,
        provider=provider,
        status_code=500,
    )


def is_retryable(exc: BaseException, *, provider: str | None = None) -> bool:
    """Return True when the error is likely transient."""

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

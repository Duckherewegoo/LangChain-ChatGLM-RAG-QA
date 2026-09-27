"""API middleware and unified exception mapping.

This module translates Python built-ins, Pydantic validation errors, and
project domain exceptions into structured responses. It must never let a raw
stack trace or secret material reach the client. Transient infrastructure
errors receive a safe generic message; a request-scoped correlation ID is
attached to every response.
"""

from __future__ import annotations

import typing
import uuid

from pydantic import BaseModel, Field

from src.exceptions import RAGError
from src.observability import get_logger, start_request

logger = get_logger(__name__)

_SAFE_INTERNAL_MESSAGE = "An unexpected internal error occurred."


class ErrorEnvelope(BaseModel):
    """Canonical API error response."""

    error: str
    message: str
    details: dict[str, typing.Any] = Field(default_factory=dict)
    request_id: str | None = None


class RequestContext(BaseModel):
    """Request correlation and tracing context."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str | None = None
    tenant_id: str | None = "default"

    def bind(self) -> None:
        """Bind this context to the current execution scope."""

        start_request(self.request_id)


def map_exception(exc: BaseException, *, request_id: str | None = None) -> tuple[int, ErrorEnvelope]:
    """Map any exception to a stable HTTP response envelope."""

    if isinstance(exc, RAGError):
        return exc.status_code, ErrorEnvelope(
            error=exc.error_code,
            message=exc.message,
            details=_safe_details(exc.details),
            request_id=request_id,
        )

    if isinstance(exc, ValueError):
        return 400, ErrorEnvelope(
            error="invalid_argument",
            message=str(exc) or "Invalid request argument.",
            request_id=request_id,
        )

    if isinstance(exc, FileNotFoundError):
        return 404, ErrorEnvelope(
            error="resource_not_found",
            message="The requested resource was not found.",
            request_id=request_id,
        )

    if isinstance(exc, PermissionError):
        return 403, ErrorEnvelope(
            error="permission_denied",
            message="The operation is not allowed for the current principal.",
            request_id=request_id,
        )

    if isinstance(exc, TimeoutError):
        return 504, ErrorEnvelope(
            error="upstream_timeout",
            message="An upstream dependency timed out.",
            request_id=request_id,
        )

    if isinstance(exc, ConnectionError):
        return 502, ErrorEnvelope(
            error="upstream_unavailable",
            message="An upstream dependency is unavailable.",
            request_id=request_id,
        )

    if isinstance(exc, NotImplementedError):
        return 501, ErrorEnvelope(
            error="not_implemented",
            message="The requested capability is not implemented.",
            request_id=request_id,
        )

    logger.exception("unexpected_api_error", extra={"request_id": request_id})
    return 500, ErrorEnvelope(
        error="internal_error",
        message=_SAFE_INTERNAL_MESSAGE,
        request_id=request_id,
    )


def ensure_request_id(existing: str | None) -> str:
    """Return the existing request ID or generate a new one."""

    return existing or uuid.uuid4().hex


def _safe_details(details: typing.Any) -> dict[str, typing.Any]:
    """Remove potentially sensitive keys from error details."""

    if not isinstance(details, dict):
        return {}
    redacted = dict(details)
    for key in list(redacted):
        lowered = key.lower()
        if any(secret in lowered for secret in ("api_key", "password", "token", "secret", "authorization")):
            redacted[key] = "[redacted]"
    return redacted


__all__ = ["ErrorEnvelope", "RequestContext", "ensure_request_id", "map_exception"]

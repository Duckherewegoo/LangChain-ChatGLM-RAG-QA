"""API 中间件与统一异常映射。

本模块将 Python 内置异常、Pydantic 校验错误以及项目领域异常翻译为结构化的
JSON 响应。它必须绝不让原始堆栈或密钥材料流向客户端。对于瞬态的基建错误，
返回安全的通用提示；每个响应都附带与当前请求关联的追踪 ID。
"""

from __future__ import annotations

import re
import typing
import uuid

from pydantic import BaseModel, Field

from src.exceptions import RAGError
from src.observability import get_logger, start_request

logger = get_logger(__name__)

_SAFE_INTERNAL_MESSAGE = "服务端发生内部错误，请稍后重试或联系管理员。"


class ErrorEnvelope(BaseModel):
    """标准的 API 错误响应体。"""

    error: str
    message: str
    details: dict[str, typing.Any] = Field(default_factory=dict)
    request_id: str | None = None


class RequestContext(BaseModel):
    """请求关联与追踪上下文。"""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str | None = None
    tenant_id: str | None = "default"

    def bind(self) -> None:
        """将当前上下文绑定到执行作用域。"""
        start_request(self.request_id)


def map_exception(exc: BaseException, *, request_id: str | None = None) -> tuple[int, ErrorEnvelope]:
    """将任意异常映射为稳定的 HTTP 响应信封（状态码 + 响应体）。"""

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
            message=str(exc) or "请求参数无效。",
            request_id=request_id,
        )

    if isinstance(exc, FileNotFoundError):
        return 404, ErrorEnvelope(
            error="resource_not_found",
            message="请求的资源未找到。",
            request_id=request_id,
        )

    if isinstance(exc, PermissionError):
        return 403, ErrorEnvelope(
            error="permission_denied",
            message="当前主体无权执行该操作。",
            request_id=request_id,
        )

    if isinstance(exc, TimeoutError):
        return 504, ErrorEnvelope(
            error="upstream_timeout",
            message="上游依赖服务响应超时。",
            request_id=request_id,
        )

    if isinstance(exc, ConnectionError):
        return 502, ErrorEnvelope(
            error="upstream_unavailable",
            message="上游依赖服务当前不可用。",
            request_id=request_id,
        )

    if isinstance(exc, NotImplementedError):
        return 501, ErrorEnvelope(
            error="not_implemented",
            message="请求的能力尚未实现。",
            request_id=request_id,
        )

    # 修复语法错误，并确保日志格式兼容
    logger.exception(f"unexpected_api_error | request_id={request_id}")
    return 500, ErrorEnvelope(
        error="internal_error",
        message=_SAFE_INTERNAL_MESSAGE,
        request_id=request_id,
    )


def ensure_request_id(existing: str | None) -> str:
    """返回现有的请求 ID，若不存在则生成一个新的。"""
    return existing or uuid.uuid4().hex


# 精确匹配真正的敏感信息字段，防止误伤诸如 max_tokens 等正常字段
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "password", "passwd", "secret", "token",
    "access_token", "refresh_token", "authorization", "credential",
})

# 匹配常见的密钥格式（如 sk-...）
_SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,})"
)


def _safe_details(details: typing.Any) -> dict[str, typing.Any]:
    """从错误详情中移除潜在的敏感信息。

    仅对明确敏感键名或符合特定密钥模式的值进行脱敏，避免误伤正常字段。
    """
    if not isinstance(details, dict):
        return {}

    redacted = dict(details)
    for key, value in list(redacted.items()):
        if key.lower() in _SENSITIVE_KEYS:
            redacted[key] = "[redacted]"
        elif isinstance(value, str):
            redacted[key] = _SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
    return redacted


__all__ = ["ErrorEnvelope", "RequestContext", "ensure_request_id", "map_exception"]

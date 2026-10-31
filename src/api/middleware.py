"""API 中间件与统一异常映射。

本模块将 Python 内置异常、Pydantic 校验错误以及项目领域异常翻译为结构化的
JSON 响应。它必须绝不让原始堆栈或密钥材料流向客户端。对于瞬态的基建错误，
返回安全的通用提示；每个响应都附带与当前请求关联的追踪 ID。

注意：`ValueError → 400` 分支只服务真正的请求参数校验；启动期配置错误应抛
`ConfigurationError`，不要用 `ValueError`，否则会被误映射为客户端错误。
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field

from base.exceptions import RAGError
from base.observability import get_logger
from base.redaction import redact_details

logger = get_logger(__name__)

_SAFE_INTERNAL_MESSAGE = "服务端发生内部错误，请稍后重试或联系管理员。"


class ErrorEnvelope(BaseModel):
    """标准的 API 错误响应体。"""

    error: str
    message: str
    details: dict[str, typing.Any] = Field(default_factory=dict)
    request_id: str | None = None


def map_exception(exc: BaseException, *, request_id: str | None = None) -> tuple[int, ErrorEnvelope]:
    """将任意异常映射为稳定的 HTTP 响应信封（状态码 + 响应体）。"""

    if isinstance(exc, RAGError):
        return exc.status_code, ErrorEnvelope(
            error=exc.error_code,
            message=exc.message,
            details=redact_details(exc.details),
            request_id=request_id,
        )

    if isinstance(exc, ValueError):
        # 仅用于真正的请求参数校验（如空 query）。
        # 启动期配置错误应抛 ConfigurationError，不要用 ValueError。
        return 400, ErrorEnvelope(
            error="invalid_argument",
            message="请求参数无效。",
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

    # FastAPI 异常处理器调用时可能已退出 except 块，显式传异常保留堆栈。
    logger.opt(exception=exc).error(f"unexpected_api_error | request_id={request_id}")
    return 500, ErrorEnvelope(
        error="internal_error",
        message=_SAFE_INTERNAL_MESSAGE,
        request_id=request_id,
    )


__all__ = ["ErrorEnvelope", "map_exception"]

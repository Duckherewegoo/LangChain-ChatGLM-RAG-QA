"""敏感信息脱敏。

递归处理 dict / list：键名按子串匹配（命中即整值 [redacted]），
字符串值按常见密钥格式做正则替换。供 api/middleware（错误响应）
与 rag/agent_events（审计事件）共用，避免两套脱敏逻辑漂移。
"""

from __future__ import annotations

import re
import typing

# 键名子串匹配；误伤 max_tokens / token_count 等可接受（宁多脱敏）。
SENSITIVE_HINTS = (
    "api_key", "apikey", "api-key", "token",
    "secret", "password", "passwd", "authorization",
    "credential", "bearer",
)

# 值正则：匹配常见密钥格式（如 sk-xxx、Bearer xxx）
SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,})"
)


def redact_details(payload: typing.Any) -> dict[str, typing.Any]:
    """递归脱敏 dict；非 dict 输入返回空 dict。"""
    if not isinstance(payload, dict):
        return {}
    return {key: _redact_value(key, value) for key, value in payload.items()}


def _redact_value(key: typing.Any, value: typing.Any) -> typing.Any:
    if isinstance(key, str) and any(hint in key.lower() for hint in SENSITIVE_HINTS):
        return "[redacted]"
    if isinstance(value, dict):
        return redact_details(value)
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
    return value


__all__ = ["SENSITIVE_HINTS", "SECRET_VALUE_PATTERN", "redact_details"]

"""HTTP API 层。

对外暴露：传输契约模型、异常映射工具。应用装配入口（create_app）通过
惰性加载获取，避免导入类型时连带触发整个应用初始化。
"""

from __future__ import annotations

import typing

from api.middleware import ErrorEnvelope, RequestContext, map_exception
from api.schemas import (
    AgentRequestBody,
    AgentResponseBody,
    AgentToolCallView,
    AnswerRequestBody,
    AnswerResponseBody,
    CitationView,
    HealthModelView,
    HealthResponseBody,
    IngestRequestBody,
    IngestResponseBody,
    MessageResponseBody,
    RetrievedDocumentView,
)

if typing.TYPE_CHECKING:
    # 仅向静态分析器声明 create_app 的存在，运行时不执行。
    from api.server import create_app


def __getattr__(name: str) -> typing.Any:
    """按需加载应用装配入口，避免导入类型时拉起重型依赖。"""
    if name == "create_app":
        from api.server import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentRequestBody",
    "AgentResponseBody",
    "AgentToolCallView",
    "AnswerRequestBody",
    "AnswerResponseBody",
    "CitationView",
    "ErrorEnvelope",
    "HealthModelView",
    "HealthResponseBody",
    "IngestRequestBody",
    "IngestResponseBody",
    "MessageResponseBody",
    "RequestContext",
    "RetrievedDocumentView",
    "create_app",
    "map_exception",
]

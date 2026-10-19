"""工具层的纯数据模型定义。"""

from __future__ import annotations

import typing
from pydantic import BaseModel, Field


class ToolContext(BaseModel):
    """传递给工具处理程序的调用上下文。

    强制不可变，防止工具处理程序在运行时篡改上下文信息。
    """

    model_config = {"frozen": True}

    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    request_id: str | None = None


class ToolSpec(BaseModel):
    """与执行后端无关的可序列化工具描述符。"""

    name: str
    description: str = ""
    input_schema: dict[str, typing.Any] = Field(default_factory=dict)
    source_type: str = "unknown"


ToolHandler = typing.Callable[[dict[str, typing.Any], ToolContext], typing.Any]

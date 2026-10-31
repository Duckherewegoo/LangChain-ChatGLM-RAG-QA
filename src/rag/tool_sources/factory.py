"""工具来源工厂。"""

from __future__ import annotations

import typing

from base.exceptions import ConfigurationError

from .base import ToolSource
from .local import LocalToolSource
from .mcp import MCPToolSource


def create_tool_source(config: dict[str, typing.Any]) -> ToolSource:
    """从配置字典构建 ToolSource。"""
    config = dict(config or {})
    source_type = str(config.pop("source_type", "local")).strip().lower()

    if source_type == "local":
        return LocalToolSource()
    if source_type == "mcp":
        command = config.get("command")
        if not command:
            raise ConfigurationError(
                "MCP 工具来源必须配置 command。",
                details={"source_type": source_type},
            )
        return MCPToolSource(
            command=command,
            args=config.get("args"),
            env=config.get("env"),
            name=config.get("name"),
            timeout_seconds=config.get("timeout_seconds", 30.0),
        )
    raise ConfigurationError(
        f"未知的工具来源类型: {source_type!r}。"
        "支持的类型为 'local' 和 'mcp'。"
    )

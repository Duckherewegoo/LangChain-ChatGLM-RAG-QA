"""Agent 注册表的可插拔工具来源。

该包提供：
- 本地 Python 工具（LocalToolSource）
- 基于 stdio 的 MCP 工具（MCPToolSource）
"""

from .base import ToolExecutionError, ToolSource
from .factory import create_tool_source
from .local import LocalToolSource
from .mcp import MCPToolSource

__all__ = [
    "LocalToolSource",
    "MCPToolSource",
    "ToolExecutionError",
    "ToolSource",
    "create_tool_source",
]

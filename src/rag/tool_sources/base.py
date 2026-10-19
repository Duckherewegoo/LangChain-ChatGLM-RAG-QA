"""工具来源抽象定义。"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod

from src.rag.tool_types import ToolContext, ToolSpec


class ToolExecutionError(RuntimeError):
    """当工具调用在来源边界失败时抛出。

    这是注册表承诺传播的唯一异常类型。
    """

    def __init__(self, message: str, *, tool: str | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.tool = tool
        self.retryable = retryable


class ToolSource(ABC):
    """抽象工具提供者。

    实现必须是轻量级的构造（__init__ 中无网络 I/O），并在连接时保持惰性：
    MCP 来源只应在首次请求列出或调用工具时启动其服务端进程。
    """

    source_type: str = "abstract"

    @abstractmethod
    def list_specs(self) -> list[ToolSpec]:
        """返回此来源暴露的工具。"""
        ...

    @abstractmethod
    def invoke(
        self,
        name: str,
        arguments: dict[str, typing.Any],
        *,
        context: ToolContext | None = None,
    ) -> typing.Any:
        """使用参数调用 name 并返回可序列化的结果。"""
        ...

    def close(self) -> None:
        """释放来源持有的任何资源。默认无操作。"""
        return

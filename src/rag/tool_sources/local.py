"""进程内工具来源。"""

from __future__ import annotations

import typing

from rag.tool_types import ToolContext, ToolHandler, ToolSpec

from .base import ToolExecutionError, ToolSource


class LocalToolSource(ToolSource):
    """基于纯 Python 可调用对象的进程内工具来源。"""

    source_type = "local"

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._schemas: dict[str, dict[str, typing.Any]] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        schema: dict[str, typing.Any],
    ) -> LocalToolSource:
        """注册本地工具。返回自身以支持链式构造。"""
        if not name or not isinstance(name, str):
            raise TypeError("工具名称必须是非空字符串。")
        if not callable(handler):
            raise TypeError(f"工具处理程序 {name!r} 必须可调用。")
        if name in self._handlers:
            raise ValueError(f"工具已注册: {name}")

        self._handlers[name] = handler
        self._schemas[name] = dict(schema)
        return self

    def names(self) -> list[str]:
        return sorted(self._handlers)

    def list_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=name,
                description=(self._schemas[name].get("description") or ""),
                input_schema=self._schemas[name],
                source_type=self.source_type,
            )
            for name in self._handlers
        ]

    def get_handler(self, name: str) -> ToolHandler:
        try:
            return self._handlers[name]
        except KeyError as exc:
            raise ToolExecutionError(
                f"本地工具 {name!r} 未注册。",
                tool=name,
                retryable=False,
            ) from exc

    def invoke(
        self,
        name: str,
        arguments: dict[str, typing.Any],
        *,
        context: ToolContext | None = None,
    ) -> typing.Any:
        handler = self.get_handler(name)
        ctx = context or ToolContext()
        return handler(arguments or {}, ctx)

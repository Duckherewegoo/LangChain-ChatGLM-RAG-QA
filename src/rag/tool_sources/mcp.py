"""基于 stdio 的 MCP 工具来源。"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import threading
import typing

from base.observability import get_logger
from rag.tool_types import ToolContext, ToolSpec

from .base import ToolExecutionError, ToolSource

logger = get_logger(__name__)


def _mcp_available() -> bool:
    return importlib.util.find_spec("mcp") is not None


class _MCPBackgroundLoop:
    """为同步环境托管的常驻 asyncio 事件循环。"""
    _instance: _MCPBackgroundLoop | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> _MCPBackgroundLoop:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        except Exception:
            logger.exception("MCP 事件循环异常退出")

    def run_coro(
        self,
        coro: typing.Coroutine[typing.Any, typing.Any, typing.Any],
        *,
        timeout: float | None = None,
    ) -> typing.Any:
        """将协程提交到常驻循环并等待结果；timeout 为 None 时无限等待。"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


class MCPToolSource(ToolSource):
    """通过 stdio 连接外部 MCP 服务端的工具来源。"""

    source_type = "mcp"

    def __init__(
        self,
        *,
        command: str,
        args: typing.Sequence[str] | None = None,
        env: dict[str, str] | None = None,
        name: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command or not isinstance(command, str):
            raise ValueError("MCPToolSource 需要非空的 command。")
        self._command = command
        self._args = list(args or [])
        self._env = dict(env or {})
        self._label = name or f"mcp:{command}"
        self._timeout = max(float(timeout_seconds), 0.1)  # 会话启动/列出/调用共用

        self._specs: list[ToolSpec] | None = None
        self._loop = _MCPBackgroundLoop.get()
        self._session = None
        self._exit_stack: contextlib.AsyncExitStack | None = None

    def _start_session_sync(self) -> None:
        if self._session is not None:
            return

        if not _mcp_available():
            raise ToolExecutionError(
                "MCP 工具来源需要 'mcp' 包。请运行 `pip install mcp`。",
                retryable=False,
            )

        from mcp import ClientSession, StdioServerParameters

        env = {**os.environ, **self._env} if self._env else None
        params = StdioServerParameters(command=self._command, args=self._args, env=env)

        async def _async_connect():
            import mcp.client.stdio as mcp_stdio
            stack = contextlib.AsyncExitStack()
            try:
                read, write = await stack.enter_async_context(mcp_stdio.stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                return session, stack
            except Exception:
                await stack.aclose()
                raise

        try:
            session, stack = self._loop.run_coro(_async_connect(), timeout=self._timeout)
            self._session = session
            self._exit_stack = stack
            logger.info(f"MCP 会话启动成功: {self._label}")
        except FileNotFoundError as exc:
            raise ToolExecutionError(f"MCP 服务端命令未找到: {self._command!r}", retryable=False) from exc
        except Exception as exc:
            raise ToolExecutionError(f"启动 MCP 服务端失败: {exc}", retryable=True) from exc

    def list_specs(self) -> list[ToolSpec]:
        """返回工具列表；首次调用后缓存，MCP 服务端变更需重建 source。"""
        if self._specs is not None:
            return self._specs

        self._start_session_sync()
        if self._session is None:
            raise ToolExecutionError(f"MCP 会话未初始化: {self._label}", retryable=True)

        async def _list():
            return await self._session.list_tools()

        try:
            result = self._loop.run_coro(_list(), timeout=self._timeout)
        except Exception as exc:
            raise ToolExecutionError(f"获取 MCP 工具列表失败: {exc}", retryable=True) from exc

        self._specs = [
            ToolSpec(
                name=tool.name,
                description=getattr(tool, "description", "") or "",
                input_schema=_tool_schema(tool),
                source_type=self.source_type,
            )
            for tool in getattr(result, "tools", [])
        ]
        return self._specs

    def invoke(
        self,
        name: str,
        arguments: dict[str, typing.Any],
        *,
        context: ToolContext | None = None,
    ) -> typing.Any:
        self._start_session_sync()
        if self._session is None:
            raise ToolExecutionError(f"MCP 会话未初始化: {self._label}", retryable=True)

        async def _call():
            return await self._session.call_tool(name, arguments or {})

        try:
            result = self._loop.run_coro(_call(), timeout=self._timeout)
        except Exception as exc:
            raise ToolExecutionError(f"MCP 工具 {name!r} 调用失败: {exc}", tool=name, retryable=True) from exc

        return _decode_mcp_result(result)

    def close(self) -> None:
        if self._session is not None and self._exit_stack is not None:
            async def _cleanup():
                try:
                    await self._exit_stack.aclose()
                except Exception as e:
                    logger.warning(f"关闭 MCP 会话时出现异常: {e}")

            # 关闭用短超时：服务端挂死时不应拖住整个关闭流程。
            # 超时/失败时仍强制清空状态字段，避免下次复用已损坏的 session。
            try:
                self._loop.run_coro(_cleanup(), timeout=5.0)
            except Exception as exc:
                logger.warning(f"MCP 会话关闭超时或失败，强制清理状态: {exc}")
            finally:
                self._session = None
                self._exit_stack = None
            logger.info(f"MCP 会话已关闭: {self._label}")

    def __repr__(self) -> str:
        return f"MCPToolSource(label={self._label!r}, command={self._command!r})"


def _tool_schema(tool) -> dict[str, typing.Any]:
    raw = getattr(tool, "inputSchema", None)
    if isinstance(raw, dict):
        return raw
    try:
        encoded = json.dumps(raw, ensure_ascii=False, default=str)
        return json.loads(encoded)
    except (TypeError, json.JSONDecodeError):
        return {}


def _decode_mcp_result(result) -> typing.Any:
    if isinstance(result, dict) and "result" in result:
        result = result["result"]

    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")

    if isinstance(content, list):
        texts = [str(getattr(p, "text", p.get("text") if isinstance(p, dict) else "")) for p in content]
        texts = [t for t in texts if t]
        joined = "\n".join(texts)
        if not joined:
            return None
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined

    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result

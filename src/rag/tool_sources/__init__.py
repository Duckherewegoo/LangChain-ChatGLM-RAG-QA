"""Pluggable tool sources for the agent registry.

A :class:`ToolSource` is anything that can advertise a set of :class:`ToolSpec`
descriptors and invoke one of them. The built-in sources are:

* :class:`LocalToolSource`   -- in-process Python callables (the original RAG tools).
* :class:`MCPToolSource`     -- tools surfaced by an `MCP`_ server over stdio.

The registry is deliberately agnostic to the source kind: from the agent's
perspective every tool is a plain callable returning JSON-serializable data.
This is the seam that lets the system grow from "three hardcoded tools" to
"three local tools plus a dozen MCP tools" without touching the agent or the
LangGraph state machine.

.. _MCP: https://modelcontextprotocol.io/
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import typing
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.rag.tools import ToolContext, ToolHandler

logger = logging.getLogger(__name__)


class ToolSpec(BaseModel):
    """A serializable tool descriptor, independent of its execution backend."""

    name: str
    description: str = ""
    schema: dict[str, typing.Any] = Field(default_factory=dict)
    source: ToolSource = Field(repr=False)

    model_config = {"arbitrary_types_allowed": True}


class ToolExecutionError(RuntimeError):
    """Raised by a source when a tool call fails at the source boundary.

    This is the *only* exception type the registry promises to propagate out of
    an MCP invocation. Source-specific errors (e.g. JSON-RPC failures) must be
    translated here so the agent never sees an SDK class it does not know.
    """

    def __init__(self, message: str, *, tool: str | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.tool = tool
        self.retryable = retryable


class ToolSource(ABC):
    """Abstract tool provider.

    Implementations must be cheap to construct (no network I/O in ``__init__``)
    and lazy about connecting: an MCP source should start its server process
    only when first asked to list or invoke tools. This keeps application boot
    fast and lets an unreachable MCP server fail at call time, not at startup.
    """

    source_type: str = "abstract"

    @abstractmethod
    def list_specs(self) -> list[ToolSpec]:
        """Return the tools exposed by this source."""

    @abstractmethod
    def invoke(
        self,
        name: str,
        arguments: dict[str, typing.Any],
        *,
        context: ToolContext | None = None,
    ) -> typing.Any:
        """Invoke ``name`` with ``arguments`` and return a serializable result."""

    def close(self) -> None:
        """Release any resources held by the source. Default: no-op."""


# ---------------------------------------------------------------------------
# Local source
# ---------------------------------------------------------------------------


class LocalToolSource(ToolSource):
    """In-process tool source backed by plain Python callables."""

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
        """Register a local tool. Returns self for fluent construction."""

        if not name or not isinstance(name, str):
            raise TypeError("Tool name must be a non-empty string.")
        if not callable(handler):
            raise TypeError(f"Tool handler {name!r} must be callable.")
        if name in self._handlers:
            raise ValueError(f"Tool already registered: {name}")
        self._handlers[name] = handler
        self._schemas[name] = schema
        return self

    def names(self) -> list[str]:
        return sorted(self._handlers)

    def list_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=name,
                description=(self._schemas[name].get("description") or ""),
                schema=self._schemas[name],
                source=self,
            )
            for name in self._handlers
        ]

    def get_handler(self, name: str) -> ToolHandler:
        try:
            return self._handlers[name]
        except KeyError as exc:
            raise ToolExecutionError(
                f"Local tool {name!r} is not registered.",
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
        if context is None:
            from src.rag.tools import ToolContext

            context = ToolContext()
        return handler(arguments or {}, context)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# MCP source
# ---------------------------------------------------------------------------


class _MCP_AVAILABLE:
    """Cached availability probe for the optional ``mcp`` dependency."""

    _checked: bool | None = None
    _client = None

    @classmethod
    def client(cls):
        if cls._client is not None:
            return cls._client
        if importlib.util.find_spec("mcp") is not None:
            from mcp import client as _client  # type: ignore[no-redef]

            cls._client = _client
        return cls._client

    @classmethod
    def available(cls) -> bool:
        if cls._checked is None:
            cls._checked = cls.client() is not None
        return cls._checked


class MCPToolSource(ToolSource):
    """Tool source backed by an `MCP`_ server.

    Connection model
    ----------------
    The source connects to exactly one MCP server, discovered via
    :class:`mcp.client.stdio.StdioServerParameters` (command + args + env).
    It is *lazy*: the subprocess and the stdio session are created on the
    first call to :meth:`list_specs` or :meth:`invoke`, never in ``__init__``.
    Each call opens a short-lived session, so the source is safe to use from
    a synchronous, request-scoped context; long-lived workloads should call
    :meth:`connect` once and reuse the source across calls.

    Failure model
    ------------
    If the ``mcp`` package is missing, construction still succeeds (no hard
    import at startup) but the first invocation raises a clear
    :class:`ToolExecutionError`. If the server process fails to start or
    returns a tool result we cannot decode, we translate the SDK exception
    instead of leaking it.

    .. _MCP: https://modelcontextprotocol.io/
    """

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
            raise ValueError("MCPToolSource requires a non-empty command.")
        self._command = command
        self._args = list(args or [])
        self._env = dict(env or {})
        self._label = name or f"mcp:{command}"
        self._timeout = max(float(timeout_seconds), 0.1)
        self._session = None
        self._owned = True
        self._specs: list[ToolSpec] | None = None

    # ---- connection lifecycle ----------------------------------------------

    def _params(self):
        from mcp import ClientSession, StdioServerParameters  # type: ignore

        return ClientSession, StdioServerParameters(
            command=self._command,
            args=self._args,
            env={**__import__("os").environ, **self._env} if self._env else None,
        )

    def _ensure_session(self):
        """Open a session if none exists. Returns (session, exit_stack)."""

        if self._session is not None:
            return self._session, None

        client = _MCP_AVAILABLE.client()
        if client is None:
            raise ToolExecutionError(
                "MCP tool source requires the 'mcp' package, which is not installed. "
                "Install it with `pip install mcp` or remove the MCP source from configuration.",
                retryable=False,
            )

        session_cls, params = self._params()
        import asyncio
        import contextlib

        stack = contextlib.AsyncExitStack()
        try:
            read, write = asyncio.run(client.stdio_client(params))
            session = asyncio.run(stack.enter_async_context(session_cls(read, write)))
            asyncio.run(session.initialize())
        except FileNotFoundError as exc:
            raise ToolExecutionError(
                f"MCP server command not found: {self._command!r}.",
                retryable=False,
            ) from exc
        except Exception as exc:
            raise ToolExecutionError(
                f"Failed to start MCP server {self._label!r}: {exc}",
                retryable=True,
            ) from exc

        self._session = session
        self._owned = False
        return session, stack

    def connect(self) -> MCPToolSource:
        """Eagerly establish the MCP session. Useful in long-lived containers."""

        self._ensure_session()
        return self

    def close(self) -> None:
        self._session = None

    # ---- ToolSource interface ----------------------------------------------

    def list_specs(self) -> list[ToolSpec]:
        if self._specs is not None:
            return self._specs

        if not _MCP_AVAILABLE.available():
            raise ToolExecutionError(
                "MCP tool source requires the 'mcp' package, which is not installed.",
                retryable=False,
            )

        import asyncio

        session, _ = self._ensure_session()
        try:
            result = asyncio.run(session.list_tools())
        except Exception as exc:
            raise ToolExecutionError(
                f"MCP list_tools failed for {self._label!r}: {exc}",
                retryable=True,
            ) from exc

        self._specs = [
            ToolSpec(
                name=tool.name,
                description=getattr(tool, "description", "") or "",
                schema=_tool_schema(tool),
                source=self,
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
        if not _MCP_AVAILABLE.available():
            raise ToolExecutionError(
                "MCP tool source requires the 'mcp' package, which is not installed.",
                retryable=False,
            )

        import asyncio

        session, _ = self._ensure_session()
        try:
            result = asyncio.run(
                session.call_tool(name, arguments or {})
            )
        except Exception as exc:
            raise ToolExecutionError(
                f"MCP tool {name!r} invocation failed: {exc}",
                tool=name,
                retryable=True,
            ) from exc

        return _decode_mcp_result(result)

    # ---- representation -----------------------------------------------------

    def __repr__(self) -> str:
        return f"MCPToolSource(label={self._label!r}, command={self._command!r})"


def _tool_schema(tool) -> dict[str, typing.Any]:
    """Extract an OpenAI-compatible JSON schema from an MCP tool object."""

    raw = getattr(tool, "inputSchema", None)
    if isinstance(raw, dict):
        return raw
    try:
        encoded = json.dumps(raw, ensure_ascii=False, default=str)
    except TypeError:
        encoded = "{}"
    return json.loads(encoded)


def _decode_mcp_result(result) -> typing.Any:
    """Flatten the MCP ``CallToolResult`` into a JSON-friendly value.

    MCP results carry a list of :class:`TextContent` parts; for a tool that
    emits a single JSON payload we return the parsed object, otherwise the
    joined text. Either way the result is always serializable so it can travel
    through the tool message back into the model context.
    """

    if isinstance(result, dict):
        result = result.get("result", result)

    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")

    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text is not None:
                texts.append(str(text))
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_tool_source(config: dict[str, typing.Any]) -> ToolSource:
    """Build a :class:`ToolSource` from a configuration dictionary.

    The dispatch key is ``source_type``. Unknown types raise immediately so a
    typo in configuration fails at load time, not when the agent first asks
    for a tool.
    """

    config = dict(config or {})
    source_type = str(config.pop("source_type", "local")).strip().lower()

    if source_type == "local":
        return LocalToolSource()
    if source_type == "mcp":
        return MCPToolSource(
            command=config["command"],
            args=config.get("args"),
            env=config.get("env"),
            name=config.get("name"),
            timeout_seconds=config.get("timeout_seconds", 30.0),
        )
    raise ValueError(
        f"Unknown tool source type: {source_type!r}. "
        "Supported values are 'local' and 'mcp'."
    )


__all__ = [
    "LocalToolSource",
    "MCPToolSource",
    "ToolExecutionError",
    "ToolSource",
    "ToolSpec",
    "create_tool_source",
]

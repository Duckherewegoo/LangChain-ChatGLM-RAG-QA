"""Agent tools, guardrails, and tool invocation records.

Tools are deliberately narrow: each handler validates its arguments, executes one
responsibility, and returns a serializable result.

Design notes
------------
* A :class:`ToolRegistry` is the **union of all registered tools**; the allow
  list is applied at ``list_tools`` / ``get`` / ``has`` time, so a registry
  built with an explicit allow list exposes exactly those tools and rejects
  every other name. This fixes the previous bug where ``ToolRegistry(["a"])``
  accepted ``"b"`` because the allow set was populated *after* registration.
* Tools come from pluggable :class:`~src.rag.tool_sources.ToolSource` providers.
  Two built-in sources exist: :class:`LocalToolSource` (the classic in-process
  handlers) and :class:`MCPToolSource` (tools surfaced by an MCP server). New
  sources are opt-in extensions, never hardcoded knowledge.
* Guardrails enforce iteration/tool budgets and detect common prompt-injection
  patterns. They are pure checks with no I/O, so they can run in tight loops.
"""

from __future__ import annotations

import importlib
import logging
import re
import typing

from pydantic import BaseModel, Field

from src.exceptions import PermissionDeniedError, ValidationError
from src.rag.tool_sources import (
    LocalToolSource,
    MCPToolSource,
    ToolSource,
    ToolSpec,
    create_tool_source,
)

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.rag.service import RAGService
    from src.settings import GuardrailsSettings

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the)\s+(rules|guidelines|prompt)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+different", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you\s+)?(said|know)", re.IGNORECASE),
]


class ToolCallRecord(BaseModel):
    """Immutable record of a tool invocation and its result."""

    tool: str
    arguments: dict[str, typing.Any] = Field(default_factory=dict)
    status: str = "success"
    result: typing.Any = None
    error: str | None = None
    source: str = "local"

    def for_audit(self) -> dict[str, typing.Any]:
        """Return a safe representation suitable for audit logs."""

        return {
            "tool": self.tool,
            "status": self.status,
            "arguments": dict(self.arguments),
            "error": self.error,
            "source": self.source,
        }


class ToolContext(BaseModel):
    """Invocation context passed to tool handlers."""

    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    request_id: str | None = None


class ToolHandler(typing.Protocol):
    """Protocol implemented by every local tool handler."""

    def __call__(self, arguments: dict[str, typing.Any], context: ToolContext) -> typing.Any: ...


def _normalize_allow_list(allowed_tools: typing.Iterable[str] | None) -> set[str] | None:
    """Translate the public API into an internal representation.

    ``None``           -> allow every currently and future-registered tool.
    ``[]``             -> allow nothing (defense in depth).
    ``["a", "b"]``     -> allow exactly that set; non-registered names are
                          *not* silently dropped, so misconfiguration is loud.
    """

    if allowed_tools is None:
        return None
    return set(allowed_tools)


class ToolRegistry:
    """Union of tool sources, with an explicit allow list and schema validation.

    The registry owns no tool implementations. It asks each configured
    :class:`ToolSource` for its specs, dedupes them by name (source order wins),
    and then answers ``list_tools`` / ``get`` / ``has`` through the allow list.
    This keeps the registry a *coordinator*, never a god object.
    """

    RETRIEVE_TOOL_NAME = "retrieve_knowledge"

    def __init__(
        self,
        sources: typing.Iterable[ToolSource] | None = None,
        *,
        allowed_tools: typing.Iterable[str] | None = None,
    ) -> None:
        self._sources: list[ToolSource] = list(sources or [])
        self._allowed = _normalize_allow_list(allowed_tools)
        self._spec_cache: dict[str, ToolSpec] | None = None

    # ---- source management (open for extension) -----------------------------

    def add_source(self, source: ToolSource) -> None:
        """Attach a tool source. Sources are queried in insertion order."""

        if not isinstance(source, ToolSource):
            raise TypeError(
                f"Tool source must implement ToolSource, got {type(source).__name__}."
            )
        self._sources.append(source)
        self._spec_cache = None

    def set_allowed(self, allowed_tools: typing.Iterable[str]) -> None:
        """Replace the allow list at runtime. ``[]`` means allow nothing."""

        self._allowed = _normalize_allow_list(allowed_tools)

    # ---- spec resolution ----------------------------------------------------

    def _resolve_specs(self) -> dict[str, ToolSpec]:
        """Return ``{name: spec}`` with first-source-wins deduplication."""

        if self._spec_cache is not None:
            return self._spec_cache
        specs: dict[str, ToolSpec] = {}
        for source in self._sources:
            for spec in source.list_specs():
                specs.setdefault(spec.name, spec)
        self._spec_cache = specs
        return specs

    def names(self) -> list[str]:
        """Every registered tool name, regardless of allow list."""

        return sorted(self._resolve_specs())

    def allowed_names(self) -> list[str]:
        """Tool names visible to the agent right now."""

        specs = self._resolve_specs()
        if self._allowed is None:
            return sorted(specs)
        return sorted(name for name in specs if name in self._allowed)

    def has(self, name: str) -> bool:
        """Return True when the tool exists *and* passes the allow list."""

        specs = self._resolve_specs()
        if name not in specs:
            return False
        if self._allowed is None:
            return True
        return name in self._allowed

    def list_tools(self) -> list[dict[str, typing.Any]]:
        """OpenAI-compatible tool definitions for the current allow list."""

        specs = self._resolve_specs()
        visible = self.allowed_names()
        return [
            {"type": "function", "function": {"name": name, **specs[name].schema}}
            for name in visible
        ]

    # ---- invocation --------------------------------------------------------

    def get(self, name: str) -> ToolHandler:
        """Return a callable for ``name``, enforcing registration and allow list.

        MCP-sourced tools are wrapped in a local callable so the execution path
        stays uniform: the agent never knows whether a tool is in-process or
        remote. This is the seam that makes MCP a pluggable extension.
        """

        specs = self._resolve_specs()
        if name not in specs:
            raise PermissionDeniedError(
                f"Unknown tool: {name!r}.",
                details={"registered_tools": sorted(specs)},
            )
        if self._allowed is not None and name not in self._allowed:
            raise PermissionDeniedError(
                f"Tool {name!r} is not permitted by the current agent configuration.",
                details={"allowed_tools": self.allowed_names()},
            )
        spec = specs[name]
        return self._wrap_spec(spec)

    def _wrap_spec(self, spec: ToolSpec) -> ToolHandler:
        """Adapt a :class:`ToolSpec` into a plain callable."""

        source = spec.source
        if isinstance(source, LocalToolSource):
            return source.get_handler(spec.name)

        def _mcp_bridge(arguments: dict[str, typing.Any], context: ToolContext) -> typing.Any:
            # MCP call sites already translate SDK exceptions into ToolExecutionError
            # or the spec's declared exception class; we do not invent a new contract.
            return source.invoke(spec.name, arguments, context=context)

        return _mcp_bridge

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release resources held by every source (MCP sessions, etc.)."""

        for source in self._sources:
            try:
                source.close()
            except Exception:
                logger.exception("tool_source_close_failed", extra={"source": type(source).__name__})

    def __enter__(self) -> ToolRegistry:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def create_rag_tool_registry(service: RAGService) -> ToolRegistry:
    """Create a registry containing retrieval and citation utilities.

    The registry owns no tool implementations. All tools are contributed by
    ``ToolSource`` providers; the registry only controls visibility through its
    allow list and exposes an ``invoke`` facade. Tests may build a registry
    with a hand-written local source to avoid touching the RAG service.
    """

    local = LocalToolSource()

    def retrieve_knowledge(
        arguments: dict[str, typing.Any], context: ToolContext
    ) -> dict[str, typing.Any]:
        from src.rag.service import AnswerRequest

        safe_arguments = dict(arguments or {})
        safe_arguments.setdefault("tenant_id", context.tenant_id)
        safe_arguments.setdefault("owner_id", context.owner_id)
        if context.allowed_sources:
            requested = safe_arguments.get("allowed_sources") or []
            merged = list(dict.fromkeys([*context.allowed_sources, *requested]))
            safe_arguments["allowed_sources"] = merged

        request = AnswerRequest(**safe_arguments)
        result = service.retrieve_only(request)
        return {
            "query": result.query,
            "document_count": len(result.documents),
            "documents": [_document_summary(document) for document in result.documents],
        }

    def list_sources(
        arguments: dict[str, typing.Any], context: ToolContext
    ) -> dict[str, typing.Any]:
        from src.rag.service import AnswerRequest

        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValidationError("list_sources requires a non-empty query.")
        result = service.retrieve_only(
            AnswerRequest(question=query, tenant_id=context.tenant_id, owner_id=context.owner_id)
        )
        return {"sources": result.distinct_sources(), "count": len(result.documents)}

    def summarize_citations(
        arguments: dict[str, typing.Any], context: ToolContext
    ) -> dict[str, typing.Any]:
        evidence = arguments.get("documents") or []
        if not isinstance(evidence, list):
            raise ValidationError("summarize_citations expects a documents array.")
        citations: list[dict[str, typing.Any]] = []
        for index, document in enumerate(evidence, start=1):
            if not isinstance(document, dict):
                continue
            citations.append(
                {
                    "index": index,
                    "source": document.get("source"),
                    "title": document.get("title"),
                    "page": document.get("page"),
                    "snippet": _snippet(document.get("content", "")),
                }
            )
        return {"citations": citations}

    local.register(
        "retrieve_knowledge",
        retrieve_knowledge,
        {
            "description": "检索企业知识库，返回受权限控制的证据片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "检索问题"},
                    "tenant_id": {"type": "string", "description": "租户标识"},
                    "owner_id": {"type": "string", "description": "限定为当前用户拥有的文档"},
                    "allowed_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "允许访问的来源白名单",
                    },
                    "top_k": {"type": "integer", "description": "最多返回的片段数"},
                },
                "required": ["question"],
            },
        },
    )
    local.register(
        "list_sources",
        list_sources,
        {
            "description": "列出当前问题命中的知识库来源。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题"},
                },
                "required": ["query"],
            },
        },
    )
    local.register(
        "summarize_citations",
        summarize_citations,
        {
            "description": "将检索结果转换为带编号的引用摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "description": "检索得到的文档对象列表",
                        "items": {"type": "object"},
                    }
                },
                "required": ["documents"],
            },
        },
    )
    # ``None`` means "allow every tool contributed by the source". This factory's
    # contract is to expose the built-in RAG tools by default; callers that need a
    # restricted set should construct their own registry or call ``set_allowed``
    # afterwards. An explicit empty sequence would mean "allow nothing", which is
    # never what the default registry wants.
    return ToolRegistry(sources=[local], allowed_tools=None)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class Guardrails:
    """Enforce iteration, tool, injection, and question-length safety rules."""

    def __init__(
        self,
        settings: GuardrailsSettings | typing.Any,
        agent_settings: typing.Any | None = None,
    ) -> None:
        self._guardrails_settings = settings
        self._agent_settings = agent_settings

    @property
    def _settings(self) -> typing.Any:
        """Backward-compatible accessor: prefer agent_settings for budget fields."""

        return self._agent_settings if self._agent_settings is not None else self._guardrails_settings

    def check_question(self, question: str) -> str:
        """Sanitize the question and reject obvious prompt injection."""

        if not isinstance(question, str) or not question.strip():
            raise ValidationError("Question must not be empty.")

        normalized = question.strip()
        if len(normalized) > self._max_question_length():
            raise ValidationError(
                f"Question exceeds maximum length of {self._max_question_length()} characters."
            )

        action = self._injection_action()
        if self._contains_injection(normalized):
            if action == "reject":
                raise ValidationError(
                    "输入包含可能覆盖系统指令的内容，已被安全策略拦截。",
                    details={"rule": "prompt_injection"},
                )
            if action == "log":
                logger.warning("Prompt injection pattern detected (logged only).")
            # "sanitize" continues after stripping matched patterns.
            for pattern in _INJECTION_PATTERNS:
                normalized = pattern.sub("", normalized).strip()
        return normalized

    def check_iteration(self, iteration: int) -> None:
        """Raise when the agent exceeds its configured iteration budget."""

        maximum = self._settings.max_iterations
        if iteration > maximum:
            raise AgentBudgetExceeded(
                f"Agent exceeded maximum iteration count: {maximum}.",
                details={"max_iterations": maximum, "iteration": iteration},
            )

    def check_tool_calls(self, count: int) -> None:
        """Raise when the agent calls too many tools."""

        maximum = self._settings.max_tool_calls
        if count > maximum:
            raise AgentBudgetExceeded(
                f"Agent exceeded maximum tool calls: {maximum}.",
                details={"max_tool_calls": maximum, "count": count},
            )

    def validate_tool_call(self, name: str, registry: ToolRegistry) -> None:
        """Ensure the requested tool exists and is allowed."""

        if not registry.has(name):
            raise PermissionDeniedError(
                f"Tool {name!r} is not registered or not permitted.",
                details={"allowed_tools": registry.allowed_names()},
            )

    def _max_question_length(self) -> int:
        configured = getattr(self._settings, "max_question_length", 0)
        return configured or 2000

    def _injection_action(self) -> str:
        return getattr(self._settings, "prompt_injection_action", "reject")

    def _contains_injection(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


class AgentBudgetExceeded(RuntimeError):
    """Raised when an agent exceeds its execution budget."""

    def __init__(self, message: str, *, details: dict[str, typing.Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


def _document_summary(document: typing.Any) -> dict[str, typing.Any]:
    return {
        "chunk_id": getattr(document, "chunk_id", None),
        "document_id": getattr(document, "document_id", None),
        "source": getattr(document, "source", None),
        "title": getattr(document, "title", None),
        "page": getattr(document, "page", None),
        "score": getattr(document, "score", None),
        "rerank_score": getattr(document, "rerank_score", None),
        "snippet": _snippet(getattr(document, "content", "")),
    }


def _snippet(text: str, limit: int = 240) -> str:
    if not isinstance(text, str):
        text = str(text)
    cleaned = " ".join(text.split())
    return cleaned[:limit] + ("…" if len(cleaned) > limit else "")


# Lazy import avoids a hard dependency on the (optional) ``mcp`` package at
# collection time: users who never enable MCP should not pay for its import.
def __getattr__(name: str):
    if name == "MCPToolSource":
        module = importlib.import_module("src.rag.tool_sources")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentBudgetExceeded",
    "Guardrails",
    "LocalToolSource",
    "MCPToolSource",
    "ToolCallRecord",
    "ToolContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolSource",
    "ToolSpec",
    "create_rag_tool_registry",
    "create_tool_source",
]

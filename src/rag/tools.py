"""Agent 工具、护栏与工具调用记录。

工具被刻意设计得极其狭隘：每个处理程序验证自己的参数，执行单一职责，并返回可序列化的结果。

设计说明：
1. ToolRegistry 是所有已注册工具的并集；白名单在 list_tools / get / has 时生效，
   因此带有显式白名单构建的注册表只会暴露那些工具，并拒绝其他所有名称。
2. 工具来自可插拔的 ToolSource。内置来源有 LocalToolSource 和 MCPToolSource。
3. 护栏强制执行迭代/工具预算，并检测常见的提示词注入模式。它们是纯检查，
   没有 I/O，因此可以在紧循环中运行。
"""

from __future__ import annotations

import logging
import re
import threading
import typing

from pydantic import BaseModel, Field

from exceptions import AgentExecutionError, PermissionDeniedError, ValidationError
from rag.tool_sources import (
    LocalToolSource,
    ToolSource,
    create_tool_source,
)
from rag.tool_types import ToolContext, ToolHandler, ToolSpec

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from rag.service import RAGService
    from settings import GuardrailsSettings

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the)\s+(rules|guidelines|prompt)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+different", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you\s+)?(said|know)", re.IGNORECASE),
]


class AgentBudgetExceeded(AgentExecutionError):
    """智能体超出执行预算时抛出。"""
    status_code = 429
    error_code = "agent_budget_exceeded"


class ToolCallRecord(BaseModel):
    """不可变的工具调用及其结果记录。"""

    model_config = {"frozen": True}

    tool: str
    arguments: dict[str, typing.Any] = Field(default_factory=dict)
    status: str = "success"
    result: typing.Any = None
    error: str | None = None
    source: str = "local"

    def for_audit(self) -> dict[str, typing.Any]:
        """返回适合审计日志的安全表示。"""
        return {
            "tool": self.tool,
            "status": self.status,
            "arguments": dict(self.arguments),
            "error": self.error,
            "source": self.source,
        }


def _normalize_allow_list(allowed_tools: typing.Iterable[str] | None) -> set[str] | None:
    """将公共 API 转换为内部表示。

    None -> 允许当前和未来注册的所有工具。
    []   -> 什么都不允许（纵深防御）。
    ["a", "b"] -> 只允许该集合。
    """
    if allowed_tools is None:
        return None
    return set(allowed_tools)


class ToolRegistry:
    """带有显式白名单和模式验证的工具来源联合体。"""

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
        self._cache_lock = threading.Lock()

    def add_source(self, source: ToolSource) -> None:
        """附加一个工具来源。来源按插入顺序查询。"""
        if not isinstance(source, ToolSource):
            raise TypeError(f"工具来源必须实现 ToolSource，实际为 {type(source).__name__}。")
        with self._cache_lock:
            self._sources.append(source)
            self._spec_cache = None

    def set_allowed(self, allowed_tools: typing.Iterable[str]) -> None:
        """运行时替换白名单。[] 表示什么都不允许。"""
        with self._cache_lock:
            self._allowed = _normalize_allow_list(allowed_tools)

    def _resolve_specs(self) -> dict[str, ToolSpec]:
        """返回 {name: spec}，首次来源优先去重。"""
        with self._cache_lock:
            if self._spec_cache is not None:
                return self._spec_cache
            specs: dict[str, ToolSpec] = {}
            for source in self._sources:
                for spec in source.list_specs():
                    specs.setdefault(spec.name, spec)
            self._spec_cache = specs
            return specs

    def names(self) -> list[str]:
        """所有已注册的工具名称，无论白名单如何。"""
        return sorted(self._resolve_specs())

    def allowed_names(self) -> list[str]:
        """智能体当前可见的工具名称。"""
        specs = self._resolve_specs()
        if self._allowed is None:
            return sorted(specs)
        return sorted(name for name in specs if name in self._allowed)

    def has(self, name: str) -> bool:
        """当工具存在且通过白名单时返回 True。"""
        specs = self._resolve_specs()
        if name not in specs:
            return False
        if self._allowed is None:
            return True
        return name in self._allowed

    def list_tools(self) -> list[dict[str, typing.Any]]:
        """当前白名单的 OpenAI 兼容工具定义。"""
        specs = self._resolve_specs()
        visible = self.allowed_names()
        return [
            {"type": "function", "function": {"name": name, **specs[name].input_schema}}
            for name in visible
        ]

    def get(self, name: str) -> ToolHandler:
        """返回 name 的可调用对象，强制执行注册和白名单检查。"""
        specs = self._resolve_specs()
        if name not in specs:
            raise PermissionDeniedError(
                f"未知工具：{name!r}。",
                details={"registered_tools": sorted(specs)},
            )
        if self._allowed is not None and name not in self._allowed:
            raise PermissionDeniedError(
                f"工具 {name!r} 不被当前智能体配置允许。",
                details={"allowed_tools": self.allowed_names()},
            )
        spec = specs[name]
        return self._wrap_spec(spec)

    def _wrap_spec(self, spec: ToolSpec) -> ToolHandler:
        """将 ToolSpec 适配为普通可调用对象。"""
        source = spec.source
        if isinstance(source, LocalToolSource):
            return source.get_handler(spec.name)

        def _mcp_bridge(arguments: dict[str, typing.Any], context: ToolContext) -> typing.Any:
            return source.invoke(spec.name, arguments, context=context)

        return _mcp_bridge

    def close(self) -> None:
        """释放每个来源持有的资源（MCP 会话等）。"""
        for source in self._sources:
            try:
                source.close()
            except Exception:
                logger.exception(f"关闭工具来源失败 | source={type(source).__name__}")

    def __enter__(self) -> ToolRegistry:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def create_rag_tool_registry(service: "RAGService") -> ToolRegistry:
    """创建一个包含检索和引用工具的注册表。"""
    local = LocalToolSource()

    def retrieve_knowledge(
        arguments: dict[str, typing.Any], context: ToolContext
    ) -> dict[str, typing.Any]:
        from rag.service import AnswerRequest

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
        from rag.service import AnswerRequest

        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValidationError("list_sources 需要非空的 query 参数。")
        result = service.retrieve_only(
            AnswerRequest(
                question=query,
                tenant_id=context.tenant_id,
                owner_id=context.owner_id,
            )
        )
        return {"sources": result.distinct_sources(), "count": len(result.documents)}

    def summarize_citations(
        arguments: dict[str, typing.Any], context: ToolContext
    ) -> dict[str, typing.Any]:
        evidence = arguments.get("documents") or []
        if not isinstance(evidence, list):
            raise ValidationError("summarize_citations 需要 documents 数组。")
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
                "properties": {"query": {"type": "string", "description": "检索问题"}},
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
    return ToolRegistry(sources=[local], allowed_tools=None)


class Guardrails:
    """强制执行迭代、工具、注入和问题长度的安全规则。"""

    def __init__(
        self,
        guardrails_settings: "GuardrailsSettings",
        agent_settings: typing.Any | None = None,
    ) -> None:
        self._guardrails_settings = guardrails_settings
        self._agent_settings = agent_settings

    @property
    def budget_settings(self) -> typing.Any:
        """预算相关的设置优先从 agent_settings 获取。"""
        return (
            self._agent_settings
            if self._agent_settings is not None
            else self._guardrails_settings
        )

    def check_question(self, question: str) -> str:
        """清理问题并拒绝明显的提示词注入。"""
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("问题不能为空。")

        normalized = question.strip()
        max_len = self._max_question_length()
        if len(normalized) > max_len:
            raise ValidationError(f"问题长度超出了 {max_len} 字符的最大限制。")

        action = self._injection_action()
        if self._contains_injection(normalized):
            if action == "reject":
                raise ValidationError(
                    "输入包含可能覆盖系统指令的内容，已被安全策略拦截。",
                    details={"rule": "prompt_injection"},
                )
            if action == "log":
                logger.warning("检测到提示词注入模式（仅记录）。")
            for pattern in _INJECTION_PATTERNS:
                normalized = pattern.sub("", normalized).strip()
        return normalized

    def check_iteration(self, iteration: int) -> None:
        """当智能体超出配置的迭代预算时抛出异常。"""
        maximum = self.budget_settings.max_iterations
        if iteration > maximum:
            raise AgentBudgetExceeded(
                f"智能体超出了最大迭代次数：{maximum}。",
                details={"max_iterations": maximum, "iteration": iteration},
            )

    def check_tool_calls(self, count: int) -> None:
        """当智能体调用工具次数过多时抛出异常。"""
        maximum = self.budget_settings.max_tool_calls
        if count > maximum:
            raise AgentBudgetExceeded(
                f"智能体超出了最大工具调用次数：{maximum}。",
                details={"max_tool_calls": maximum, "count": count},
            )

    def validate_tool_call(self, name: str, registry: ToolRegistry) -> None:
        """确保请求的工具存在且被允许。"""
        if not registry.has(name):
            raise PermissionDeniedError(
                f"工具 {name!r} 未注册或未被允许。",
                details={"allowed_tools": registry.allowed_names()},
            )

    def _max_question_length(self) -> int:
        configured = getattr(self._guardrails_settings, "max_question_length", 0)
        return configured or 2000

    def _injection_action(self) -> str:
        return getattr(self._guardrails_settings, "prompt_injection_action", "reject")

    def _contains_injection(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


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


__all__ = [
    "AgentBudgetExceeded",
    "Guardrails",
    "LocalToolSource",
    "ToolCallRecord",
    "ToolContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolSource",
    "ToolSpec",
    "create_rag_tool_registry",
    "create_tool_source",
]

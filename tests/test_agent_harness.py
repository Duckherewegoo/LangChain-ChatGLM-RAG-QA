"""Agent harness、审计轨迹与 Fallback 可见性测试。"""

from __future__ import annotations

from rag.agent_events import AgentAuditTrail, AgentEventType
from rag.tool_sources import LocalToolSource
from rag.tool_types import ToolContext
from rag.tools import ToolRegistry


class _StubModel:
    """最小可用的模型 stub，只实现协议要求的方法。"""

    name = "stub"

    def __init__(self, response):
        self._response = response

    def invoke(self, request):
        return self._response

    async def ainvoke(self, request):
        return self._response

    def bind_tools(self, tools):
        return self


def _event_type_str(value: str) -> str:
    """把 AgentEventType 成员或字符串统一为字符串。

    兼容 AgentEventType 从普通类迁移到枚举的过程。
    """
    if isinstance(value, str):
        return value
    return getattr(value, "value", str(value))


def _retrieval_tool_registry():
    local = LocalToolSource()

    def retrieve(arguments, context: ToolContext):
        # 用 context.tenant_id 显式做隔离，让这个测试名副其实
        visible = context.tenant_id == "tenant-a"
        return {
            "query": arguments.get("question"),
            "document_count": 1 if visible else 0,
            "documents": [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "source": "manual.pdf",
                    "title": "手册",
                    "score": 0.9,
                    "content": "年假需提前申请。",
                }
            ] if visible else [],
        }

    local.register(
        "retrieve_knowledge",
        retrieve,
        {
            "description": "retrieve",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    )
    registry = ToolRegistry()
    registry.add_source(local)
    return registry


def test_tool_registry_enforces_tenant_context():
    """租户上下文确实被传到了 handler，且 handler 据此过滤结果。"""
    registry = _retrieval_tool_registry()
    allowed = registry.get("retrieve_knowledge")(
        {"question": "如何申请年假"},
        ToolContext(tenant_id="tenant-a", owner_id="user-1"),
    )
    denied = registry.get("retrieve_knowledge")(
        {"question": "如何申请年假"},
        ToolContext(tenant_id="tenant-b", owner_id="user-2"),
    )
    assert allowed["document_count"] == 1
    assert denied["document_count"] == 0


def test_tool_registry_rejects_unknown_tool():
    """白名单外的工具名不会出现在注册表中。"""
    local = LocalToolSource()
    local.register(
        "retrieve_knowledge",
        lambda arguments, context: {"document_count": 0},
        {
            "description": "retrieve",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    )
    registry = ToolRegistry(allowed_tools=["retrieve_knowledge"])
    registry.add_source(local)
    assert registry.has("retrieve_knowledge") is True
    assert registry.has("rmrf") is False


def test_audit_trail_records_and_summarizes():
    """审计轨迹能聚合模型调用、fallback 与工具事件。"""
    trail = AgentAuditTrail(request_id="req-1")
    trail.record_now(
        event_type=_event_type_str(AgentEventType.MODEL_CALL),
        model="glm-4",
        provider="zhipu",
    )
    trail.record_now(
        event_type=_event_type_str(AgentEventType.MODEL_FALLBACK),
        provider="deepseek",
        status="ok",
    )
    trail.record_now(
        event_type=_event_type_str(AgentEventType.TOOL_CALL_SUCCEEDED),
        tool="retrieve_knowledge",
    )
    summary = trail.summarize()

    assert summary["event_count"] == 3
    assert summary["fallback_count"] == 1
    assert summary["tools"] == ["retrieve_knowledge"]
    assert "zhipu" in summary["providers"]
    assert "deepseek" in summary["providers"]


def test_audit_event_redacts_secrets():
    """审计事件序列化时脱敏敏感字段。"""
    trail = AgentAuditTrail(request_id="req-2")
    event = trail.record_now(
        event_type=_event_type_str(AgentEventType.MODEL_CALL),
        data={"api_key": "sk-secret", "safe": "value"},
    )
    serialized = event.for_audit()
    assert serialized["data"]["api_key"] == "[redacted]"
    assert serialized["data"]["safe"] == "value"

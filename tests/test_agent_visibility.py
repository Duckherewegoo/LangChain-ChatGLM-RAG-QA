"""Tests for the agent harness, audit trail, and fallback visibility."""

from __future__ import annotations

from src.rag.agent_events import AgentAuditTrail, AgentEventType
from src.rag.tools import ToolRegistry, ToolContext
from src.rag.tool_sources import LocalToolSource


class _StubModel:
    name = "stub"

    def __init__(self, response):
        self._response = response

    def invoke(self, request):
        return self._response

    async def ainvoke(self, request):
        return self._response

    def bind_tools(self, tools):
        return self


def _retrieval_tool_registry():
    local = LocalToolSource()

    def retrieve(arguments, context: ToolContext):
        return {
            "query": arguments.get("question"),
            "document_count": 1,
            "documents": [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "source": "manual.pdf",
                    "title": "手册",
                    "score": 0.9,
                    "content": "年假需提前申请。",
                }
            ],
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
    registry = _retrieval_tool_registry()
    context = ToolContext(tenant_id="tenant-a", owner_id="user-1")
    result = registry.get("retrieve_knowledge")({"question": "如何申请年假"}, context)
    assert result["document_count"] == 1


def test_tool_registry_rejects_unknown_tool():
    local = LocalToolSource()
    local.register(
        "retrieve_knowledge",
        lambda arguments, context: {"document_count": 0},
        {"description": "retrieve"},
    )
    registry = ToolRegistry(allowed_tools=["retrieve_knowledge"])
    registry.add_source(local)
    assert registry.has("retrieve_knowledge") is True
    assert registry.has("rmrf") is False


def test_audit_trail_records_and_summarizes():
    trail = AgentAuditTrail(request_id="req-1")
    trail.record_now(event_type=AgentEventType.MODEL_CALL, model="glm-4", provider="zhipu")
    trail.record_now(
        event_type=AgentEventType.MODEL_FALLBACK,
        provider="deepseek",
        status="ok",
    )
    trail.record_now(
        event_type=AgentEventType.TOOL_CALL_SUCCEEDED,
        tool="retrieve_knowledge",
    )
    summary = trail.summarize()

    assert summary["event_count"] == 3
    assert summary["fallback_count"] == 1
    assert summary["tools"] == ["retrieve_knowledge"]
    assert "zhipu" in summary["providers"] or "deepseek" in summary["providers"]


def test_audit_event_redacts_secrets():
    trail = AgentAuditTrail(request_id="req-2")
    event = trail.record_now(
        event_type=AgentEventType.MODEL_CALL,
        data={"api_key": "sk-secret", "safe": "value"},
    )
    serialized = event.for_audit()
    assert serialized["data"]["api_key"] == "[redacted]"
    assert serialized["data"]["safe"] == "value"

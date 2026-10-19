"""智能体执行事件与审计轨迹。

每一次图状态转移、工具调用、模型调用和最终回答都被表示为一个不可变事件。
事件可以安全地序列化，并保留了足够的上下文以供审查，同时不会暴露原始的工具负载或凭证。
"""

from __future__ import annotations

import enum
import json
import time
import typing
import uuid

from pydantic import BaseModel, Field


class AgentEventType(str, enum.Enum):
    """稳定的事件类别标识符。"""

    STARTED = "agent.started"
    ITERATION_STARTED = "agent.iteration.started"
    REASONING = "agent.reasoning"
    TOOL_CALL_REQUESTED = "agent.tool_call.requested"
    TOOL_CALL_SUCCEEDED = "agent.tool_call.succeeded"
    TOOL_CALL_FAILED = "agent.tool_call.failed"
    RETRIEVAL = "agent.retrieval"
    CONTEXT_UPDATED = "agent.context.updated"
    GUARDRAIL_TRIGGERED = "agent.guardrail.triggered"
    REVIEW_STARTED = "agent.review.started"
    REVIEW_COMPLETED = "agent.review.completed"
    REVIEW_ISSUE = "agent.review.issue"
    REVIEW_OVERRIDDEN = "agent.review.overridden"
    ROLE_DISPATCHED = "agent.role.dispatched"
    MODEL_CALL = "agent.model.call"
    MODEL_FALLBACK = "agent.model.fallback"
    FINALIZED = "agent.finalized"
    COMPLETED = "agent.completed"
    FAILED = "agent.failed"


class AgentEvent(BaseModel):
    """智能体执行期间发出的单条可审计事件。"""

    model_config = {"frozen": True}  # 强制不可变

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: str
    request_id: str
    timestamp: float = Field(default_factory=time.time)
    iteration: int = 0
    node: str | None = None
    tool: str | None = None
    model: str | None = None
    provider: str | None = None
    status: str = "ok"
    message: str | None = None
    data: dict[str, typing.Any] = Field(default_factory=dict)
    latency_seconds: float | None = None
    retryable: bool = False

    def for_audit(self) -> dict[str, typing.Any]:
        """返回一个脱敏后的 JSON 安全字典。"""
        redacted = _redact(dict(self.data or {}))
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "node": self.node,
            "tool": self.tool,
            "model": self.model,
            "provider": self.provider,
            "status": self.status,
            "message": self.message,
            "latency_seconds": self.latency_seconds,
            "retryable": self.retryable,
            "data": redacted,
        }


class AgentAuditTrail(BaseModel):
    """有序的智能体事件序列。"""

    request_id: str
    events: list[AgentEvent] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    
    # 内部监听器列表，不参与序列化
    _listeners: list[AgentEventListener] = []

    def add_listener(self, listener: "AgentEventListener") -> None:
        """注册事件监听器。"""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def record(self, event: AgentEvent) -> None:
        """追加事件，并通知所有已注册的监听器。"""
        self.events.append(event)
        for listener in self._listeners:
            try:
                listener.on_event(event)
            except Exception:
                # 监听器报错不应阻断主业务流程
                pass

    def record_now(self, **fields: typing.Any) -> AgentEvent:
        """创建、追加并返回一个事件。"""
        event = AgentEvent(request_id=self.request_id, **fields)
        self.record(event)
        return event

    def complete(self) -> None:
        """将轨迹标记为已完成。"""
        self.completed_at = time.time()

    def duration_seconds(self) -> float | None:
        """当轨迹完成时返回总耗时。"""
        if self.completed_at is None:
            return None
        return max(0.0, self.completed_at - self.started_at)

    def summarize(self) -> dict[str, typing.Any]:
        """返回适合 API 响应的紧凑摘要。"""
        models = _collect_unique(self.events, "model")
        providers = _collect_unique(self.events, "provider")
        tools = _collect_unique(self.events, "tool")
        failures = [event for event in self.events if event.status == "error"]
        fallback_events = [event for event in self.events if event.event_type == AgentEventType.MODEL_FALLBACK.value]
        return {
            "request_id": self.request_id,
            "event_count": len(self.events),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds(),
            "models": models,
            "providers": providers,
            "tools": tools,
            "iterations": _max_iteration(self.events),
            "failure_count": len(failures),
            "fallback_count": len(fallback_events),
            "failure_types": sorted({event.event_type for event in failures}),
        }

    def to_jsonl(self) -> str:
        """将每一个事件序列化为 JSON Lines。"""
        return "\n".join(
            json.dumps(event.for_audit(), ensure_ascii=False) for event in self.events
        )


class AgentEventListener:
    """智能体轨迹的观察者接口。"""

    def on_event(self, event: AgentEvent) -> None:
        """处理被发出的事件。"""
        raise NotImplementedError


class LoggingAgentEventListener(AgentEventListener):
    """通过标准日志记录器持久化事件。"""

    def __init__(self, logger: typing.Any, *, only_types: typing.Iterable[str] | None = None) -> None:
        self._logger = logger
        self._only = set(only_types) if only_types else None

    def on_event(self, event: AgentEvent) -> None:
        if self._only and event.event_type not in self._only:
            return
        
        # 兼容标准 logging 库，使用 extra 传递结构化字段
        extra_info = event.for_audit()
        self._logger.info(
            f"agent_event | type={event.event_type} | node={event.node or 'N/A'}", agent_event=extra_info)


def _redact(payload: dict[str, typing.Any]) -> dict[str, typing.Any]:
    cleaned: dict[str, typing.Any] = {}
    sensitive = {"api_key", "token", "secret", "password", "authorization", "credential"}
    for key, value in payload.items():
        if isinstance(key, str) and key.lower() in sensitive:
            cleaned[key] = "[redacted]"
        elif isinstance(value, dict):
            cleaned[key] = _redact(value)
        elif isinstance(value, list):
            cleaned[key] = [_redact(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


def _collect_unique(events: typing.Iterable[AgentEvent], attribute: str) -> list[str]:
    values: list[str] = []
    for event in events:
        value = getattr(event, attribute)
        if value and value not in values:
            values.append(value)
    return values


def _max_iteration(events: typing.Iterable[AgentEvent]) -> int:
    maximum = 0
    for event in events:
        maximum = max(maximum, event.iteration)
    return maximum


__all__ = [
    "AgentAuditTrail",
    "AgentEvent",
    "AgentEventListener",
    "AgentEventType",
    "LoggingAgentEventListener",
]

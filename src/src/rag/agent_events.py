"""Agent execution events and audit trails.

Every graph transition, tool invocation, model call, and final answer is
represented as an immutable event. Events are safe to serialize and retain
enough context for review without exposing raw tool payloads or credentials.
"""

from __future__ import annotations

import time
import typing
import uuid

from pydantic import BaseModel, Field


class AgentEventType(str):
    """Stable event category identifiers."""

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
    """A single auditable event emitted during agent execution."""

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
        """Return a JSON-safe copy with redacted sensitive fields."""

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
    """Ordered, immutable sequence of agent events."""

    request_id: str
    events: list[AgentEvent] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None

    def record(self, event: AgentEvent) -> None:
        """Append an event; the trail remains the only event history."""

        self.events.append(event)

    def record_now(self, **fields: typing.Any) -> AgentEvent:
        """Create, append, and return an event."""

        event = AgentEvent(request_id=self.request_id, **fields)
        self.record(event)
        return event

    def complete(self) -> None:
        """Mark the trail as completed."""

        self.completed_at = time.time()

    def duration_seconds(self) -> float | None:
        """Return total elapsed time when the trail is complete."""

        if self.completed_at is None:
            return None
        return max(0.0, self.completed_at - self.started_at)

    def summarize(self) -> dict[str, typing.Any]:
        """Return a compact summary suitable for API responses."""

        models = _collect_unique(self.events, "model")
        providers = _collect_unique(self.events, "provider")
        tools = _collect_unique(self.events, "tool")
        failures = [event for event in self.events if event.status == "error"]
        fallback_events = [event for event in self.events if event.event_type == AgentEventType.MODEL_FALLBACK]
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
        """Serialize every event as JSON Lines."""

        return "\n".join(
            __import__("json").dumps(event.for_audit(), ensure_ascii=False) for event in self.events
        )


class AgentEventListener:
    """Observer interface for agent trails."""

    def on_event(self, event: AgentEvent) -> None:
        """Handle an emitted event."""

        raise NotImplementedError


class LoggingAgentEventListener(AgentEventListener):
    """Persist events through the structured logger."""

    def __init__(self, logger: typing.Any, *, only_types: typing.Iterable[str] | None = None) -> None:
        self._logger = logger
        self._only = set(only_types) if only_types else None

    def on_event(self, event: AgentEvent) -> None:
        if self._only and event.event_type not in self._only:
            return
        log = self._logger.bind(request_id=event.request_id, event_type=event.event_type)
        log.info("agent_event", **event.for_audit())


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

"""API transport schemas.

These models define the JSON contract exposed to HTTP clients. They deliberately
duplicate only the fields required for transport instead of reusing internal
domain models, keeping the API backward-compatible and independent of internal
refactoring.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field


class AnswerRequestBody(BaseModel):
    """Transport model for synchronous RAG requests."""

    question: str = Field(..., min_length=1, max_length=4000)
    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=100)
    history: list[dict[str, str]] = Field(default_factory=list)
    use_agent: bool = False
    additional_context: str | None = None
    model: str | None = Field(default=None, description="Optional named model from model_router.models")
    use_case: str = Field(default="chat", description="Routing use case, e.g. chat or agent")


class AgentRequestBody(BaseModel):
    """Transport model for agent requests."""

    question: str = Field(..., min_length=1, max_length=4000)
    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    additional_context: str | None = None
    use_case: str = "agent"


class IngestRequestBody(BaseModel):
    """Transport model for document ingestion."""

    path: str
    tenant_id: str = "default"
    owner_id: str | None = None
    title: str | None = None
    tags: list[str] = Field(default_factory=list)


class CitationView(BaseModel):
    """API representation of a source citation."""

    index: int
    source: str
    title: str = ""
    page: int | None = None
    chunk_id: str = ""


class RetrievedDocumentView(BaseModel):
    """API representation of a retrieved document."""

    chunk_id: str
    source: str
    title: str = ""
    page: int | None = None
    score: float = 0.0
    rerank_score: float | None = None
    tenant_id: str = "default"


class AnswerResponseBody(BaseModel):
    """Synchronous answer response."""

    answer: str
    query: str
    citations: list[CitationView] = Field(default_factory=list)
    documents: list[RetrievedDocumentView] = Field(default_factory=list)
    context_used: bool = False
    model: str | None = None
    tokens: dict[str, int | None] = Field(default_factory=dict)
    attempted_models: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    request_id: str | None = None


class AgentToolCallView(BaseModel):
    """API representation of one agent tool invocation."""

    tool: str
    status: str
    arguments: dict[str, typing.Any] = Field(default_factory=dict)
    error: str | None = None


class AgentResponseBody(BaseModel):
    """Agent execution response."""

    request_id: str
    answer: str
    citations: list[dict[str, typing.Any]] = Field(default_factory=list)
    tool_calls: list[AgentToolCallView] = Field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "completed"
    model: str | None = None
    documents: list[dict[str, typing.Any]] = Field(default_factory=list)
    attempted_models: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class IngestResponseBody(BaseModel):
    """Document ingestion response."""

    status: str
    message: str
    document_id: str | None = None
    scanned: int = 0
    chunks_stored: int = 0
    bytes_size: int = 0
    request_id: str | None = None


class HealthModelView(BaseModel):
    """Health status of one model provider."""

    provider: str
    model_name: str
    configured: bool = False
    reachable: bool = False
    latency_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    supports_tools: bool = False


class HealthResponseBody(BaseModel):
    """Service health response."""

    status: str
    app_env: str
    api_configured: bool = False
    vector_store: str
    collection: str
    embedding_model: str
    embedding_dimension: int = 0
    collection_compatible: bool = True
    models: list[HealthModelView] = Field(default_factory=list)
    primary_model: str | None = None
    available_models: list[str] = Field(default_factory=list)
    degraded: bool = False


class MessageResponseBody(BaseModel):
    """Generic operation response."""

    status: str
    message: str
    request_id: str | None = None


__all__ = [
    "AgentRequestBody",
    "AgentResponseBody",
    "AgentToolCallView",
    "AnswerRequestBody",
    "AnswerResponseBody",
    "CitationView",
    "HealthModelView",
    "HealthResponseBody",
    "IngestRequestBody",
    "IngestResponseBody",
    "MessageResponseBody",
    "RetrievedDocumentView",
]

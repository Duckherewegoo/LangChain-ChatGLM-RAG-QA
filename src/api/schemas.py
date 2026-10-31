"""API 传输层模型。

这些模型定义了对 HTTP 客户端暴露的 JSON 契约。它们刻意只复制传输所需的字段，
而不复用内部领域模型，从而保持 API 向后兼容，并独立于内部重构。

校验分层：本层只做结构校验（类型、非空）；业务校验（问题长度上限等）
由 Guardrails 在领域层完成，避免 schema 与配置出现两个长度上限。
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field


class AnswerRequestBody(BaseModel):
    """同步 RAG 请求的传输模型。"""

    question: str = Field(..., min_length=1)
    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=100)
    history: list[dict[str, str]] = Field(default_factory=list)
    additional_context: str | None = Field(default=None, description="预留：当前 handler 未消费")
    model: str | None = Field(default=None, description="预留：当前由服务端配置决定，未消费")
    use_case: str = Field(default="chat", description="路由用例；当前仅出现在错误详情，不参与路由")


class AgentRequestBody(BaseModel):
    """Agent 请求的传输模型。"""

    question: str = Field(..., min_length=1)
    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    additional_context: str | None = Field(default=None, description="预留：handler 会告警并丢弃")
    use_case: str = Field(default="agent", description="预留：handler 当前丢弃，未消费")


class IngestRequestBody(BaseModel):
    """文档摄入的传输模型。"""

    path: str
    tenant_id: str = "default"
    owner_id: str | None = None
    title: str | None = None
    tags: list[str] = Field(default_factory=list)


class CitationView(BaseModel):
    """来源引用的 API 表示。"""

    index: int
    source: str
    title: str = ""
    page: int | None = None
    chunk_id: str = ""


class RetrievedDocumentView(BaseModel):
    """检索文档的 API 表示。"""

    document_id: str = ""
    chunk_id: str
    source: str
    title: str = ""
    page: int | None = None
    score: float = 0.0
    rerank_score: float | None = None
    tenant_id: str = "default"
    snippet: str = ""


class AnswerResponseBody(BaseModel):
    """同步回答响应。"""

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
    """单次 Agent 工具调用的 API 表示。"""

    tool: str
    status: str
    arguments: dict[str, typing.Any] = Field(default_factory=dict)
    error: str | None = None


class AgentResponseBody(BaseModel):
    """Agent 执行响应。"""

    request_id: str
    answer: str
    citations: list[CitationView] = Field(default_factory=list)
    tool_calls: list[AgentToolCallView] = Field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "completed"
    model: str | None = None
    documents: list[RetrievedDocumentView] = Field(default_factory=list)
    attempted_models: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class IngestResponseBody(BaseModel):
    """文档摄入响应。"""

    status: str
    message: str
    document_id: str | None = None
    scanned: int = 0
    chunks_stored: int = 0
    bytes_size: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list, description="逐文件失败明细；空表示全部成功")
    request_id: str | None = None


class HealthModelView(BaseModel):
    """单个模型提供者的健康状态。"""

    provider: str
    model_name: str
    configured: bool = False
    reachable: bool = False
    latency_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    supports_tools: bool = False


class HealthResponseBody(BaseModel):
    """服务健康检查响应。"""

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
    """通用操作响应。"""

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

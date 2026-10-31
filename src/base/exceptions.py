"""RAG 服务的领域异常定义。

这些异常用于区分配置、校验、模型、检索和智能体故障，
以便应用层能够选择适当的 HTTP 状态码和可观测性动作，
而无需解析错误字符串。
"""

from __future__ import annotations

from typing import Any


class RAGError(Exception):
    """所有项目错误的基类。"""

    status_code: int = 500
    error_code: str = "rag_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.trace_id = trace_id

    def to_dict(self) -> dict[str, Any]:
        """将异常转换为标准字典，供 API 层直接返回。

        注意：details 可能含模型名、尝试记录等内部标识，对外暴露前应由
        api/middleware 按需脱敏。
        """
        payload = {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }
        if self.trace_id:
            payload["trace_id"] = self.trace_id
        return payload

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(RAGError):
    """应用配置无效或不完整。"""
    status_code = 500
    error_code = "configuration_error"


class ValidationError(RAGError):
    """请求负载未通过业务校验。"""
    status_code = 400
    error_code = "validation_error"


class ModelNotConfiguredError(RAGError):
    """无法根据当前配置构建有效的模型客户端。配置缺失是永久性错误。"""
    status_code = 500
    error_code = "model_not_configured"


class ModelConfigurationError(ModelNotConfiguredError):
    """模型提供者配置无效（缺少 API Key、端点非法等）。"""
    status_code = 500
    error_code = "model_configuration_error"


class ModelInvocationError(RAGError):
    """模型调用在配置的重试次数后仍然失败。"""
    status_code = 502
    error_code = "model_invocation_error"


class DocumentLoadError(RAGError):
    """文档无法打开或解码。"""
    status_code = 400
    error_code = "document_load_error"


class IngestError(RAGError):
    """文档摄入流水线失败。"""
    status_code = 500
    error_code = "ingest_error"


class RetrievalError(RAGError):
    """向量检索或重排失败。"""
    status_code = 502
    error_code = "retrieval_error"


class AgentExecutionError(RAGError):
    """智能体因触发守卫或不可恢复的工具错误而停止。"""
    status_code = 502
    error_code = "agent_execution_error"


class ToolExecutionError(RAGError):
    """智能体调用的工具执行失败（如参数不合法、外部 API 崩溃）。"""
    status_code = 500
    error_code = "tool_execution_error"


class PermissionDeniedError(RAGError):
    """用户或租户无权访问请求的资源。"""
    status_code = 403
    error_code = "permission_denied"


class ResourceNotFoundError(RAGError):
    """请求的租户、集合、文档或会话未找到。"""
    status_code = 404
    error_code = "resource_not_found"


class CollectionCompatibilityError(RAGError):
    """向量集合与当前 Embedding 模型不兼容（维度不一致等）。"""
    status_code = 409
    error_code = "collection_incompatible"


__all__ = [
    "AgentExecutionError",
    "CollectionCompatibilityError",
    "ConfigurationError",
    "DocumentLoadError",
    "IngestError",
    "ModelConfigurationError",
    "ModelInvocationError",
    "ModelNotConfiguredError",
    "PermissionDeniedError",
    "RAGError",
    "ResourceNotFoundError",
    "RetrievalError",
    "ToolExecutionError",
    "ValidationError",
]

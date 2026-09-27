"""Domain exceptions for the RAG service.

These exceptions distinguish configuration, validation, model, retrieval, and
agent failures so that application layers can choose appropriate HTTP status
codes and observability actions without inspecting error strings.
"""

from __future__ import annotations

from typing import Any


class RAGError(Exception):
    """Base class for all project errors."""

    status_code: int = 500
    error_code: str = "rag_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.error_code, "message": self.message, "details": self.details}


class ConfigurationError(RAGError):
    """Invalid or incomplete application configuration."""

    status_code = 500
    error_code = "configuration_error"


class ValidationError(RAGError):
    """Request payload failed business validation."""

    status_code = 400
    error_code = "validation_error"


class ModelNotConfiguredError(RAGError):
    """No valid model client can be constructed from current settings."""

    status_code = 503
    error_code = "model_not_configured"


class ModelInvocationError(RAGError):
    """A model call failed after configured retries."""

    status_code = 502
    error_code = "model_invocation_error"


class DocumentLoadError(RAGError):
    """A document could not be opened or decoded."""

    status_code = 400
    error_code = "document_load_error"


class IngestError(RAGError):
    """Document ingestion pipeline failed."""

    status_code = 500
    error_code = "ingest_error"


class RetrievalError(RAGError):
    """Vector retrieval or reranking failed."""

    status_code = 502
    error_code = "retrieval_error"


class AgentExecutionError(RAGError):
    """The agent stopped because of guardrails or an unrecoverable tool error."""

    status_code = 502
    error_code = "agent_execution_error"


class PermissionDeniedError(RAGError):
    """The user or tenant is not allowed to access the requested resource."""

    status_code = 403
    error_code = "permission_denied"


class ResourceNotFoundError(RAGError):
    """A requested tenant, collection, document, or conversation was not found."""

    status_code = 404
    error_code = "resource_not_found"


__all__ = [
    "AgentExecutionError",
    "ConfigurationError",
    "DocumentLoadError",
    "IngestError",
    "ModelInvocationError",
    "ModelNotConfiguredError",
    "PermissionDeniedError",
    "RAGError",
    "ResourceNotFoundError",
    "RetrievalError",
    "ValidationError",
]

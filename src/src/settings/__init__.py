"""Application configuration.

This module loads configuration from environment variables, an optional ``.env``
file, and an optional YAML document. Environment values take precedence so that
secret management and deployment automation do not require changing committed
configuration files.
"""

from __future__ import annotations

import os
import typing

import yaml
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Routing and embedding types live in a leaf module (routing_types) to avoid
# circular imports.  We import them eagerly here and bind them into this
# module's globals so that Pydantic can resolve the string annotations used by
# AppSettings at class-creation time.
from src.model.routing_types import ModelEmbeddingSettings, ModelRouterSettings
from src.settings.yaml_loader import YamlConfigSource

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

globals()["ModelRouterSettings"] = ModelRouterSettings
globals()["ModelEmbeddingSettings"] = ModelEmbeddingSettings

_ENV_FILE = os.environ.get("APP_SETTINGS_ENV_FILE", ".env")
_YAML_FILE = os.environ.get("APP_SETTINGS_YAML_FILE", "config/settings.yaml")
_UNKNOWN_REPR = "<masked>"


class ModelChatSettings(BaseModel):
    """Chat model connection parameters."""

    provider: str = Field(default="zhipu", description="Model provider identifier.")
    model_name: str = Field(default="glm-5.2", description="Model identifier selected by the provider.")
    api_key: str = Field(default="", description="API key for the selected provider.")
    base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="OpenAI-compatible endpoint base URL.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int | None = None
    timeout_seconds: float = Field(default=45.0, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        if not value:
            return value
        return value.rstrip("/") + "/"

    @property
    def is_configured(self) -> bool:
        """Whether a non-empty API key has been supplied."""

        return bool(self.api_key.strip())


class VectorStoreSettings(BaseModel):
    """Vector store backend settings."""

    type: str = Field(default="chroma", description="Vector store backend type.")
    collection: str = Field(default="enterprise_knowledge")
    persist_directory: str = Field(default="./data/index")

class RerankSettings(BaseModel):
    """Reranking configuration."""

    enabled: bool = True
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    top_k: int = Field(default=4, ge=1, le=50)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)




class RetrievalSettings(BaseModel):
    """Retrieval pipeline settings."""

    top_k: int = Field(default=8, ge=1, le=100)
    fetch_multiplier: int = Field(default=3, ge=1, le=10)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)


class IngestSettings(BaseModel):
    """Document ingestion settings."""

    supported_extensions: list[str] = Field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".docx"]
    )
    chunk_size: int = Field(default=600, ge=50, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    min_characters: int = Field(default=50, ge=1, le=1000)

    @field_validator("chunk_overlap")
    @classmethod
    def _validate_overlap(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size", 600)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        return value


class ReviewSettings(BaseModel):
    """Answer reviewer configuration.

    Controls whether the Reviewer role runs after the Reasoner, and how many
    times the orchestrator may ask the Reasoner to retry when the Reviewer
    returns blocking issues. ``use_llm`` decides whether the reviewer attempts
    semantic review first; when disabled or the model call fails, the reviewer
    falls back to deterministic rule-based checks.
    """

    enabled: bool = True
    use_llm: bool = False
    max_retries: int = Field(default=1, ge=0, le=3)
    max_evidence_chars: int = Field(default=6000, ge=256, le=60000)


class AgentSettings(BaseModel):
    """Agent runtime settings."""

    max_iterations: int = Field(default=6, ge=1, le=50)
    max_tool_calls: int = Field(default=8, ge=1, le=100)
    allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "retrieve_knowledge",
            "list_sources",
            "summarize_citations",
        ]
    )
    review: ReviewSettings = Field(default_factory=ReviewSettings)


class GuardrailsSettings(BaseModel):
    """Security and input validation settings.

    This is the single authoritative definition of guardrail configuration.
    """

    max_question_length: int = Field(default=2000, ge=1, le=100000)
    max_history_messages: int = Field(default=12, ge=0, le=200)
    prompt_injection_action: str = Field(default="reject")

    @field_validator("prompt_injection_action")
    @classmethod
    def _validate_action(cls, value: str) -> str:
        allowed = {"reject", "sanitize", "log"}
        if value not in allowed:
            raise ValueError(f"prompt_injection_action must be one of {allowed}, got {value!r}.")
        return value


class ObservabilitySettings(BaseModel):
    """Observability (logging, metrics, tracing) settings."""

    tracing_enabled: bool = False
    metrics_enabled: bool = True


class ApiSettings(BaseModel):
    """HTTP API server settings."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


def _ModelEmbeddingSettings() -> ModelEmbeddingSettings:
    """Return default embedding settings."""

    return ModelEmbeddingSettings()


def _ModelRouterSettings() -> ModelRouterSettings:
    """Return default router configuration."""

    return ModelRouterSettings()


class AppSettings(BaseSettings):
    """Root application configuration object."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "enterprise-rag-chatglm"
    app_env: str = "development"
    model_chat: ModelChatSettings = Field(default_factory=ModelChatSettings)
    model_embedding: ModelEmbeddingSettings = Field(default_factory=_ModelEmbeddingSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    guardrails: GuardrailsSettings = Field(default_factory=GuardrailsSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    model_router: ModelRouterSettings = Field(default_factory=_ModelRouterSettings)

    def describe(self) -> dict[str, object]:
        """Return a safe-to-log summary of the configuration."""

        return {
            "app_name": self.app_name,
            "app_env": self.app_env,
            "chat_model": self.model_chat.model_name,
            "chat_provider": self.model_chat.provider,
            "embedding_model": self.model_embedding.model_name,
            "collection": self.retrieval.vector_store.collection,
            "api_host": self.api.host,
            "api_port": self.api.port,
        }


class ConfigurationError(RuntimeError):
    """Raised when application configuration is incomplete or invalid."""


def load_settings(
    yaml_path: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> AppSettings:
    """Load settings from YAML and environment with precedence rules.

    Environment variables take precedence over YAML values. Secrets should be
    supplied via environment variables or a secrets manager.
    """

    path = yaml_path or _YAML_FILE
    env = environment if environment is not None else os.environ

    document: dict[str, object] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ConfigurationError(f"YAML root at {path} must be a mapping.")
        document = dict(loaded)

    merged = _apply_environment(document, env)
    try:
        return AppSettings(**merged)
    except ValidationError as exc:  # pragma: no cover - re-raised as domain error
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc


def _apply_environment(
    document: dict[str, object], environment: Mapping[str, str]
) -> dict[str, object]:
    """Merge environment-variable overrides into the YAML document."""

    merged = dict(document)

    chat_overrides = _extract_section_overrides(environment, "MODEL_CHAT")
    embed_overrides = _extract_section_overrides(environment, "MODEL_EMBED")
    ingest_overrides = _extract_section_overrides(environment, "INGEST")
    retrieval_overrides = _extract_section_overrides(environment, "RETRIEVAL")
    vector_overrides = _extract_section_overrides(environment, "RETRIEVAL_VECTOR")
    rerank_overrides = _extract_section_overrides(environment, "RETRIEVAL_RERANK")

    if chat_overrides:
        merged["model_chat"] = _deep_merge(dict(merged.get("model_chat") or {}), chat_overrides)
    if embed_overrides:
        merged["model_embedding"] = _deep_merge(
            dict(merged.get("model_embedding") or {}), embed_overrides
        )
    if ingest_overrides:
        merged["ingest"] = _deep_merge(dict(merged.get("ingest") or {}), ingest_overrides)
    if retrieval_overrides or vector_overrides:
        existing: dict[str, object] = dict(merged.get("retrieval") or {})
        if vector_overrides:
            existing["vector_store"] = _deep_merge(
                dict(existing.get("vector_store") or {}), vector_overrides
            )
        if retrieval_overrides:
            existing = _deep_merge(existing, retrieval_overrides)
        merged["retrieval"] = existing
    if rerank_overrides:
        merged["rerank"] = _deep_merge(dict(merged.get("rerank") or {}), rerank_overrides)

    return merged


def _extract_section_overrides(
    environment: Mapping[str, str], prefix: str
) -> dict[str, object]:
    """Extract environment variables matching a section prefix."""

    result: dict[str, object] = {}
    prefix_upper = prefix.upper()
    for key, value in environment.items():
        if key.startswith(prefix_upper + "_"):
            field = key[len(prefix_upper) + 1 :].lower()
            result[field] = _coerce_env_value(value)
    return result


def _coerce_env_value(value: str) -> object:
    """Coerce an environment variable string to its appropriate Python type."""

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _deep_merge(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    """Recursively merge two mappings, giving precedence to ``overrides``."""

    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


__all__ = [
    "AgentSettings",
    "AppSettings",
    "ConfigurationError",
    "GuardrailsSettings",
    "IngestSettings",
    "ModelChatSettings",
    "ObservabilitySettings",
    "RerankSettings",
    "RetrievalSettings",
    "VectorStoreSettings",
    "load_settings",
    "YamlConfigSource",
]

# Finalize AppSettings schema: the routing/embedding types were injected into
# this module's globals() at import time, so Pydantic can now resolve the
# string-based forward annotations used by AppSettings fields.
AppSettings.model_rebuild()

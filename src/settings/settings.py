"""Application configuration.

Loads configuration from environment variables, an optional ``.env`` file,
and an optional YAML document. Environment values take precedence so that
secret management and deployment automation do not require changing
committed configuration files.
"""

from __future__ import annotations

import os
import typing

import yaml
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.model.routing_types import ModelEmbeddingSettings, ModelRouterSettings
from src.settings.yaml_loader import YamlConfigSource

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

globals()["ModelRouterSettings"] = ModelRouterSettings
globals()["ModelEmbeddingSettings"] = ModelEmbeddingSettings


class ModelChatSettings(BaseModel):
    """Settings for a single LLM endpoint."""

    provider: str = Field(..., description="Provider identifier, e.g. openai, zhipu")
    model_name: str = Field(..., description="Model identifier")
    api_key: str = Field("", description="API key; loaded from env in production")
    base_url: str | None = Field(None, description="Optional custom endpoint")
    timeout_seconds: float = Field(30.0, ge=0)
    max_retries: int = Field(2, ge=0)


class VectorStoreSettings(BaseModel):
    """Vector store connection settings."""

    backend: str = Field("faiss", description="faiss | milvus | chroma")
    persist_dir: str = Field("./data/index", description="On-disk index location")
    dimension: int = Field(0, ge=0, description="Embedding dimension; 0 = auto-detect")


class RerankSettings(BaseModel):
    """Reranking stage configuration."""

    enabled: bool = False
    model: str = ""
    top_k: int = Field(20, ge=1)


class RetrievalSettings(BaseModel):
    """Top-level retrieval tunables."""

    top_k: int = Field(5, ge=1)
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    hybrid_weight: float = Field(0.5, ge=0.0, le=1.0)
    query_rewrite: str = Field("expansion", description="identity | expansion | llm")
    rerank: RerankSettings = Field(default_factory=RerankSettings)


class IngestSettings(BaseModel):
    """Document ingestion configuration."""

    chunk_size: int = Field(512, ge=1)
    chunk_overlap: int = Field(64, ge=0)
    batch_size: int = Field(32, ge=1)


class ReviewSettings(BaseModel):
    """Answer review stage configuration."""

    enabled: bool = True
    use_llm: bool = False
    max_retries: int = Field(1, ge=0)
    max_evidence_chars: int = Field(4000, ge=0)


class AgentSettings(BaseModel):
    """Agent loop and role configuration."""

    max_iterations: int = Field(8, ge=1)
    timeout_seconds: float = Field(60.0, ge=0)
    review: ReviewSettings = Field(default_factory=ReviewSettings)


class GuardrailsSettings(BaseModel):
    """This is the single, authoritative definition of GuardrailsSettings."""

    max_question_length: int = Field(2000, ge=1)
    max_answer_length: int = Field(8000, ge=1)


class ObservabilitySettings(BaseModel):
    """Logging and metrics configuration."""

    metrics_enabled: bool = True
    log_level: str = Field("INFO", description="DEBUG | INFO | WARNING | ERROR")


class ApiSettings(BaseModel):
    """HTTP server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class AppSettings(BaseSettings):
    """Root configuration object assembled from all sources."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )

    models: dict[str, ModelChatSettings] = Field(default_factory=dict)
    default_model: str = "chat"
    model_embedding: "ModelEmbeddingSettings"
    model_router: "ModelRouterSettings"
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    guardrails: GuardrailsSettings = Field(default_factory=GuardrailsSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    def model_post_init(self, __context: object) -> None:
        # routing_types factories are called here, after all fields exist, so
        # we never access a partially-initialized object.
        if not hasattr(self, "model_embedding") or self.model_embedding is None:
            self.model_embedding = ModelEmbeddingSettings()
        if not hasattr(self, "model_router") or self.model_router is None:
            self.model_router = ModelRouterSettings()


class ConfigurationError(RuntimeError):
    """Raised when settings cannot be assembled or validated."""


def load_settings(
    config_path: str | os.PathLike[str] | None = None,
    *,
    overrides: typing.Mapping[str, typing.Any] | None = None,
) -> AppSettings:
    """Load and validate application settings.

    Resolution order (later wins): built-ins → YAML file → environment →
    explicit ``overrides`` dict. Raises :class:`ConfigurationError` with the
    underlying :class:`ValidationError` chained when validation fails.
    """

    sources: list[dict[str, typing.Any]] = [_DEFAULT_CONFIG]

    yaml_path = config_path or os.environ.get("RAG_CONFIG")
    if yaml_path:
        try:
            sources.append(YamlConfigSource(yaml_path).load())
        except FileNotFoundError as exc:
            raise ConfigurationError(f"config file not found: {yaml_path}") from exc

    env_overrides = _extract_section_overrides(os.environ)
    if env_overrides:
        sources.append(env_overrides)
    if overrides:
        sources.append(dict(overrides))

    merged = _deep_merge(*sources)
    try:
        return AppSettings(**merged)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc


_DEFAULT_CONFIG: dict[str, typing.Any] = {
    "models": {},
    "default_model": "chat",
    "retrieval": {},
    "vector_store": {},
    "ingest": {},
    "agent": {},
    "guardrails": {},
    "observability": {},
    "api": {},
}


def _apply_environment(
    env: typing.Mapping[str, str],
) -> dict[str, typing.Any]:
    """Translate flat environment variables into the nested shape AppSettings expects."""

    result: dict[str, typing.Any] = {}
    prefix = "RAG_"

    for key, raw in env.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        cursor: dict[str, typing.Any] = result
        for segment in parts[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[parts[-1]] = _coerce_env_value(raw)

    return result


def _extract_section_overrides(
    env: typing.Mapping[str, str],
) -> dict[str, typing.Any]:
    """Return a nested override dict built from RAG_-prefixed variables."""

    return _apply_environment(env)


def _coerce_env_value(value: str) -> typing.Any:
    """Best-effort conversion of a string env value to a Python primitive."""

    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.isdigit():
        return int(value)
    if value.replace(".", "", 1).isdigit():
        return float(value)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [s.strip() for s in inner.split(",")] if inner else []
    return value


def _deep_merge(
    base: dict[str, typing.Any], *overrides: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """Recursively merge mapping overrides without mutating the base."""

    result = dict(base)
    for override in overrides:
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
    return result


__all__ = [
    "AppSettings",
    "ConfigurationError",
    "load_settings",
    "ModelChatSettings",
    "VectorStoreSettings",
    "RetrievalSettings",
    "RerankSettings",
    "IngestSettings",
    "AgentSettings",
    "GuardrailsSettings",
    "ObservabilitySettings",
    "ApiSettings",
]

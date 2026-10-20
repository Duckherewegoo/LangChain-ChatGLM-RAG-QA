"""应用配置

从环境变量、可选的 .env 文件和可选的 YAML 文档加载配置。
环境变量的值优先级最高，因此密钥管理和部署自动化无需修改已提交的配置文件。
"""

from __future__ import annotations

import json
import os
import typing
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from model.routing_types import ModelEmbeddingSettings, ModelRouterSettings


class ModelChatSettings(BaseModel):
    """单个 LLM 端点的配置。"""

    provider: str = Field(default="", description="提供者标识，如 openai / zhipu / deepseek")
    model_name: str = Field(default="", description="模型标识")
    api_key: str = Field(default="", description="API Key；生产环境从环境变量注入")
    base_url: str | None = Field(default=None, description="可选的自定义端点")
    temperature: float = Field(default=0.2, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @property
    def is_configured(self) -> bool:
        """三者齐全才视为可用。"""
        return bool(self.provider and self.model_name and self.api_key)


class VectorStoreSettings(BaseModel):
    """向量存储连接配置。"""

    type: str = Field(default="chromadb", description="chromadb | faiss | milvus")
    collection: str = Field(default="enterprise_knowledge")
    collection_strategy: str = Field(default="model_fingerprint")
    max_collection_name_length: int = Field(default=63, ge=1)
    persist_directory: str = Field(default="./data/index")
    hnsw_space: str = Field(default="cosine")
    dimension: int = Field(default=0, ge=0, description="0 表示自动探测")


class RerankSettings(BaseModel):
    """重排阶段配置。"""

    enabled: bool = False
    model_name: str = ""
    top_k: int = Field(default=20, ge=1)


class KeywordSettings(BaseModel):
    """关键词检索通道配置。"""

    enabled: bool = False
    top_k: int = Field(default=20, ge=1)


class FusionSettings(BaseModel):
    """多路结果融合配置。"""

    type: str = Field(default="rrf", description="rrf | weighted | dedupe")
    k: int = Field(default=60, ge=1)
    weights: dict[str, float] = Field(default_factory=dict)


class RetrievalSettings(BaseModel):
    """检索阶段的顶层可调参数。"""

    top_k: int = Field(default=5, ge=1)
    fetch_multiplier: int = Field(default=3, ge=1)
    per_query_top_k: int = Field(default=20, ge=1)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    hybrid_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    query_rewrite: str = Field(default="expansion", description="identity | expansion | llm")
    rerank: RerankSettings = Field(default_factory=lambda: RerankSettings())
    keyword: KeywordSettings = Field(default_factory=lambda: KeywordSettings())
    fusion: FusionSettings = Field(default_factory=lambda: FusionSettings())
    vector_store: VectorStoreSettings = Field(default_factory=lambda: VectorStoreSettings())


class IngestSettings(BaseModel):
    """文档摄入配置。"""

    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=64, ge=0)
    min_characters: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=1)
    supported_extensions: list[str] = Field(
        default_factory=lambda: ["txt", "md", "json", "pdf", "docx"]
    )


class ReviewSettings(BaseModel):
    """答案审核阶段配置。"""

    enabled: bool = True
    use_llm: bool = False
    max_retries: int = Field(default=1, ge=0)
    max_evidence_chars: int = Field(default=4000, ge=0)


class AgentSettings(BaseModel):
    """Agent 循环与角色配置。"""

    max_iterations: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=16, ge=1)
    timeout_seconds: float = Field(default=60.0, ge=0)
    review: ReviewSettings = Field(default_factory=lambda: ReviewSettings())


class GuardrailsSettings(BaseModel):
    """安全护栏配置（唯一权威定义）。

    预算相关的 max_iterations / max_tool_calls 归属 AgentSettings，
    本类只负责注入防御与长度限制。
    """

    max_question_length: int = Field(default=2000, ge=1)
    max_answer_length: int = Field(default=8000, ge=1)
    prompt_injection_action: str = Field(
        default="reject", description="reject | log | sanitize"
    )


class ObservabilitySettings(BaseModel):
    """日志与指标配置。"""

    metrics_enabled: bool = True
    log_level: str = Field(default="INFO", description="DEBUG | INFO | WARNING | ERROR")


class ApiSettings(BaseModel):
    """HTTP 服务配置。"""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])


class AppSettings(BaseModel):
    """从所有来源合并出的根配置对象。

    使用 BaseModel 而非 BaseSettings，因为 load_settings 已经全权负责
    YAML、环境变量和 overrides 的优先级合并，避免 Pydantic Settings 再读一遍
    环境变量造成双重读取和优先级歧义。
    """

    app_env: str = Field(default="dev", description="dev | staging | prod")
    log_level: str = Field(default="INFO", description="DEBUG | INFO | WARNING | ERROR")

    model_chat: ModelChatSettings = Field(default_factory=lambda: ModelChatSettings())
    model_embedding: ModelEmbeddingSettings = Field(
        default_factory=lambda: ModelEmbeddingSettings()
    )
    model_router: ModelRouterSettings = Field(default_factory=lambda: ModelRouterSettings())

    retrieval: RetrievalSettings = Field(default_factory=lambda: RetrievalSettings())
    ingest: IngestSettings = Field(default_factory=lambda: IngestSettings())
    agent: AgentSettings = Field(default_factory=lambda: AgentSettings())
    guardrails: GuardrailsSettings = Field(default_factory=lambda: GuardrailsSettings())
    observability: ObservabilitySettings = Field(
        default_factory=lambda: ObservabilitySettings()
    )
    api: ApiSettings = Field(default_factory=lambda: ApiSettings())


class ConfigurationError(RuntimeError):
    """配置装配或校验失败时抛出。"""


def load_settings(
    config_path: str | os.PathLike[str] | None = None,
    *,
    overrides: typing.Mapping[str, typing.Any] | None = None,
) -> AppSettings:
    sources: list[dict[str, typing.Any]] = []

    yaml_path = _resolve_yaml_path(config_path)
    if yaml_path is not None:
        raw_yaml = _load_yaml_safely(yaml_path)
        sources.append(_normalize_yaml_config(raw_yaml))

    # 提取并归一化环境变量
    env_overrides = _normalize_env_config(os.environ)
    if env_overrides:
        sources.append(env_overrides)

    # 兼容原有的 RAG_ 前缀写法（给未来留口子，保留你原本的设计）
    rag_overrides = _apply_environment(os.environ)
    if rag_overrides:
        sources.append(rag_overrides)

    if overrides:
        sources.append(dict(overrides))

    merged = _deep_merge({}, *sources)

    try:
        return AppSettings(**merged)
    except ValidationError as exc:
        raise ConfigurationError(f"配置校验失败: {exc}") from exc

def _resolve_yaml_path(config_path: str | os.PathLike[str] | None) -> Path | None:
    """按显式参数 → 环境变量 → 项目默认路径 → CWD 的顺序解析 YAML 路径。

    显式指定的路径不存在时抛错；默认路径不存在时返回 None。
    """
    if config_path:
        candidate = Path(config_path).expanduser()
        if candidate.exists():
            return candidate
        raise ConfigurationError(f"配置路径不存在: {candidate}")

    env_path = os.environ.get("RAG_CONFIG") or os.environ.get("APP_SETTINGS_YAML_FILE")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate
        raise ConfigurationError(f"RAG_CONFIG 指定的文件不存在: {candidate}")

    project_root = Path(__file__).resolve().parent.parent.parent
    candidate = project_root / "config" / "settings.yaml"
    if candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / "config" / "settings.yaml"
    if cwd_candidate.exists():
        return cwd_candidate

    return None


def _load_yaml_safely(path: Path) -> dict[str, typing.Any]:
    """读取 YAML 文件；解析失败或顶层非 dict 时抛出 ConfigurationError。"""
    try:
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"YAML 配置解析失败 | path={path} | error={exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"YAML 文件读取失败 | path={path} | error={exc}") from exc

    if not isinstance(document, dict):
        raise ConfigurationError(
            f"YAML 顶层必须是映射（dict），实际为 {type(document).__name__} | path={path}"
        )
    return document


def _normalize_yaml_config(raw: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """将 YAML 中的习惯写法（app, model.chat）归一化为 AppSettings 的扁平字段。"""
    result = dict(raw)
    
    # 1. 适配 app -> app_name, app_env, api
    app_cfg = result.pop("app", {})
    if isinstance(app_cfg, dict):
        if "name" in app_cfg:
            result.setdefault("app_name", app_cfg["name"])
        if "env" in app_cfg:
            result.setdefault("app_env", app_cfg["env"])
        api = result.setdefault("api", {})
        if "host" in app_cfg:
            api.setdefault("host", app_cfg["host"])
        if "port" in app_cfg:
            api.setdefault("port", app_cfg["port"])

    # 2. 适配 model.chat / model.embedding
    model_cfg = result.pop("model", {})
    if isinstance(model_cfg, dict):
        if "chat" in model_cfg:
            result.setdefault("model_chat", model_cfg["chat"])
        if "embedding" in model_cfg:
            result.setdefault("model_embedding", model_cfg["embedding"])

    # 3. 适配 security -> guardrails
    security_cfg = result.pop("security", {})
    if isinstance(security_cfg, dict):
        result.setdefault("guardrails", security_cfg)

    return result


def _normalize_env_config(env: typing.Mapping[str, str]) -> dict[str, typing.Any]:
    """将 .env 中的扁平键名映射到 AppSettings 的嵌套字段。"""
    result: dict[str, typing.Any] = {}
    
    def _set_nested(keys: list[str], value: typing.Any) -> None:
        cursor = result
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[keys[-1]] = value

    for key, val in env.items():
        # 1. 处理通用前缀
        if key == "APP_ENV": _set_nested(["app_env"], val)
        elif key == "APP_NAME": _set_nested(["app_name"], val)
        elif key == "APP_LOG_LEVEL": _set_nested(["log_level"], val)
        elif key == "APP_HOST": _set_nested(["api", "host"], val)
        elif key == "APP_PORT": _set_nested(["api", "port"], _coerce_env_value(val))

        # 2. 处理 MODEL_CHAT_
        elif key.startswith("MODEL_CHAT_"):
            field = key[len("MODEL_CHAT_"):].lower()
            _set_nested(["model_chat", field], _coerce_env_value(val))

        # 3. 处理 MODEL_EMBED_
        elif key.startswith("MODEL_EMBED_"):
            field = key[len("MODEL_EMBED_"):].lower()
            _set_nested(["model_embedding", field], _coerce_env_value(val))

        # 4. 处理 RETRIEVAL_
        elif key.startswith("RETRIEVAL_"):
            field = key[len("RETRIEVAL_"):].lower()
            # 处理嵌套，如 RETRIEVAL_RERANK_ENABLED -> retrieval.rerank.enabled
            if field.startswith("rerank_"):
                _set_nested(["retrieval", "rerank", field[7:]], _coerce_env_value(val))
            elif field.startswith("vector_store_"):
                _set_nested(["retrieval", "vector_store", field[13:]], _coerce_env_value(val))
            else:
                _set_nested(["retrieval", field], _coerce_env_value(val))

        # 5. 处理 AGENT_
        elif key.startswith("AGENT_"):
            field = key[len("AGENT_"):].lower()
            if field == "allowed_tools":
                # 特殊处理：把逗号分隔字符串转为列表
                _set_nested(["agent", field], [s.strip() for s in val.split(",") if s.strip()])
            else:
                _set_nested(["agent", field], _coerce_env_value(val))

        # 6. 处理 SECURITY_ -> guardrails
        elif key.startswith("SECURITY_"):
            field = key[len("SECURITY_"):].lower()
            _set_nested(["guardrails", field], _coerce_env_value(val))

    return result

def _apply_environment(env: typing.Mapping[str, str]) -> dict[str, typing.Any]:
    """将 RAG_ 前缀的扁平环境变量翻译成 AppSettings 期望的嵌套结构。

    例如 RAG_MODEL_CHAT__API_KEY=sk-xxx 会变成
        {"model_chat": {"api_key": "sk-xxx"}}
    """
    result: dict[str, typing.Any] = {}
    prefix = "RAG_"

    for key, raw in env.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        if not parts or not parts[0]:
            continue
        cursor: dict[str, typing.Any] = result
        for segment in parts[:-1]:
            nxt = cursor.setdefault(segment, {})
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[segment] = nxt
            cursor = nxt
        cursor[parts[-1]] = _coerce_env_value(raw)

    return result


def _coerce_env_value(value: str) -> typing.Any:
    """将字符串环境变量尽量还原为 Python 原生类型。

    优先按 JSON 解析（覆盖 int、float、bool、null、list、dict）；
    失败则尝试解析 [a,b,c] 形式的列表；再失败则保留为字符串。
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip() for item in inner.split(",")]

    return value


def _deep_merge(
    base: dict[str, typing.Any], *overrides: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """递归合并且不修改 base；嵌套字典会逐层合并，非字典值直接覆盖。"""
    result = dict(base)
    for override in overrides:
        if not isinstance(override, dict):
            continue
        for key, value in override.items():
            existing = result.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                result[key] = _deep_merge(existing, value)
            else:
                result[key] = value
    return result


__all__ = [
    "AgentSettings",
    "ApiSettings",
    "AppSettings",
    "ConfigurationError",
    "FusionSettings",
    "GuardrailsSettings",
    "IngestSettings",
    "KeywordSettings",
    "ModelChatSettings",
    "ModelEmbeddingSettings",
    "ModelRouterSettings",
    "ObservabilitySettings",
    "RerankSettings",
    "RetrievalSettings",
    "ReviewSettings",
    "VectorStoreSettings",
    "load_settings",
]

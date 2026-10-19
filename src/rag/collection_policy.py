"""向量集合策略与嵌入维度治理。

不同的 Embedding 模型会产生不兼容的向量几何结构。本模块为每个模型分配一个持久化的指纹，
据此推导出唯一的集合名称，并拒绝打开维度与当前 Embedder 不一致的现有集合。
这可以防止运维人员在未重建索引的情况下替换 Embedding 模型所导致的静默数据损坏。
"""

from __future__ import annotations

import hashlib
import re
import typing

from pydantic import BaseModel

from src.exceptions import ConfigurationError, IngestError

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from src.model.embedding import EmbeddingModel
    from src.settings import VectorStoreSettings

_FINGERPRINT_VERSION = 1
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_\-]")


class EmbeddingFingerprint(BaseModel):
    """Embedding 配置的稳定身份标识。"""

    model_name: str
    dimension: int
    normalize: bool
    version: int = _FINGERPRINT_VERSION
    digest: str = ""

    @property
    def short_id(self) -> str:
        """用于集合名称的紧凑哈希片段。"""
        if self.digest:
            return self.digest[:12]
        return self.compute_digest()[:12]

    def compute_digest(self) -> str:
        """计算指纹的规范 SHA-256 摘要。"""
        payload = f"v{self.version}|{self.model_name}|{self.dimension}|{self.normalize}".encode()
        return hashlib.sha256(payload).hexdigest()

    def with_digest(self) -> EmbeddingFingerprint:
        """返回填充了摘要的副本。"""
        return EmbeddingFingerprint(
            model_name=self.model_name,
            dimension=self.dimension,
            normalize=self.normalize,
            version=self.version,
            digest=self.compute_digest(),
        )


class CollectionPolicy(BaseModel):
    """将 Embedding 指纹映射到物理集合的规则。"""

    base_name: str = "enterprise_knowledge"
    strategy: str = "model_fingerprint"
    max_name_length: int = 63

    def resolve(self, fingerprint: EmbeddingFingerprint) -> str:
        """返回当前 Embedding 模型对应的集合名称。"""
        if self.strategy == "fixed":
            return self._safe(self.base_name)

        if self.strategy == "model_name":
            return self._safe(f"{self.base_name}_{self._slug(fingerprint.model_name)}")

        suffix = f"{fingerprint.dimension}d_{fingerprint.short_id}"
        return self._safe(f"{self.base_name}_{suffix}")

    def _safe(self, name: str) -> str:
        """清洗非法字符，并在超长时优先保留唯一后缀。"""
        cleaned = _SAFE_NAME.sub("_", name).strip("_")
        if not cleaned:
            raise ConfigurationError("解析出的集合名称为空。")

        if len(cleaned) <= self.max_name_length:
            return cleaned

        # 安全截断：保留首部前缀和尾部的唯一哈希标识
        half_len = self.max_name_length // 2
        prefix = cleaned[:half_len]
        suffix = cleaned[-(self.max_name_length - half_len):]
        return f"{prefix}_{suffix}"

    @staticmethod
    def _slug(text: str) -> str:
        return _SAFE_NAME.sub("_", text).strip("_").lower() or "unknown"


class CollectionCompatibilityError(IngestError):
    """当现有集合与当前 Embedder 不兼容时抛出。"""

    status_code = 409
    error_code = "collection_incompatible"


def _extract_normalize(embedder: typing.Any) -> bool:
    """优先从公共属性读取 normalize，回退到 _settings，再回退到 True。"""
    normalize = getattr(embedder, "normalize", None)
    if normalize is not None:
        return bool(normalize)

    settings = getattr(embedder, "_settings", None)
    if settings is not None:
        value = getattr(settings, "normalize", None)
        if value is not None:
            return bool(value)

    return True


def fingerprint_embedder(embedder: EmbeddingModel, model_name: str) -> EmbeddingFingerprint:
    """从 Embedder 实例创建指纹。"""
    dimension = getattr(embedder, "dimension", 0)
    if not dimension:
        raise ConfigurationError(
            "Embedder 缺乏可测量的维度；在打开集合前必须先加载模型。",
            details={"model_name": model_name},
        )

    return EmbeddingFingerprint(
        model_name=model_name,
        dimension=dimension,
        normalize=_extract_normalize(embedder),
    ).with_digest()


def create_collection_policy(settings: VectorStoreSettings) -> CollectionPolicy:
    """创建应用配置中声明的策略。"""
    return CollectionPolicy(
        base_name=settings.collection,
        strategy=settings.collection_strategy,
        max_name_length=settings.max_collection_name_length,
    )


def validate_compatibility(
    policy: CollectionPolicy,
    fingerprint: EmbeddingFingerprint,
    stored_dimension: int | None,
) -> str:
    """返回集合名称，如果维度不兼容则抛出异常。"""
    collection = policy.resolve(fingerprint)
    if stored_dimension is None:
        return collection

    if stored_dimension != fingerprint.dimension:
        raise CollectionCompatibilityError(
            "现有集合的维度与当前 Embedding 模型不匹配。",
            details={
                "collection": collection,
                "stored_dimension": stored_dimension,
                "active_dimension": fingerprint.dimension,
                "fingerprint": fingerprint.short_id,
            },
        )
    return collection


def describe_compatibility(
    policy: CollectionPolicy,
    fingerprint: EmbeddingFingerprint,
    stored_dimension: int | None,
) -> dict[str, typing.Any]:
    """返回非抛出异常的兼容性摘要，用于诊断。"""
    if stored_dimension is None:
        return {
            "collection": policy.resolve(fingerprint),
            "compatible": True,
            "status": "empty",
            "active_dimension": fingerprint.dimension,
            "stored_dimension": None,
        }

    compatible = stored_dimension == fingerprint.dimension
    return {
        "collection": policy.resolve(fingerprint),
        "compatible": compatible,
        "status": "compatible" if compatible else "incompatible",
        "active_dimension": fingerprint.dimension,
        "stored_dimension": stored_dimension,
        "fingerprint": fingerprint.short_id,
    }


__all__ = [
    "CollectionCompatibilityError",
    "CollectionPolicy",
    "EmbeddingFingerprint",
    "create_collection_policy",
    "describe_compatibility",
    "fingerprint_embedder",
    "validate_compatibility",
]

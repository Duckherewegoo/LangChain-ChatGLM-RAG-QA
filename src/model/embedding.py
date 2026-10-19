"""Embedding abstraction and default local implementation.

定义 Embedding 模型协议接口与默认的本地 Sentence-Transformers 实现。
业务层通过该模块定义的接口进行操作，隔离具体的底层实现。
"""

from __future__ import annotations

import hashlib
import threading
import typing

from pydantic import BaseModel

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from collections.abc import Sequence

    from sentence_transformers import SentenceTransformer

    from src.model.routing_types import ModelEmbeddingSettings

_E5_QUERY_INSTRUCTION = "query: "


@typing.runtime_checkable
class EmbeddingModel(typing.Protocol):
    """Embedding 模型协议接口。

    具体实现类不需要显式继承本类，只要实现了以下属性和方法即可。
    """

    name: str
    model_name: str
    dimension: int
    normalize: bool

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """将文档文本转换为向量表示。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """将搜索查询转换为向量表示。"""
        ...


class _LocalEmbeddingModel:
    """基于 Sentence-Transformers 的本地 Embedding 实现。

    特性：
    1. 懒加载底层模型。
    2. 自动清理空白字符与畸形输入。
    3. 为 E5 系列模型自动添加查询前缀。
    """

    def __init__(self, settings: ModelEmbeddingSettings) -> None:
        self.name = "local"
        self._settings = settings
        self.model_name = settings.model_name
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self.dimension = 0
        self.normalize = settings.normalize

    def _load(self) -> SentenceTransformer:
        """线程安全的懒加载机制。"""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._settings.model_name, device=self._settings.device)
            self._model = model
            self.dimension = int(model.get_sentence_embedding_dimension())
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量处理文档向量化。"""
        prepared = [_normalize_text(text) for text in texts]
        if not prepared:
            return []

        model = self._load()
        vectors = model.encode(
            prepared,
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self._settings.normalize,
            convert_to_numpy=True,
        )
        return [list(map(float, row)) for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        """处理查询向量化，兼容 E5 模型前缀。"""
        cleaned = _normalize_text(text)
        if not cleaned:
            raise ValueError("不能对空查询进行向量化处理。")

        model = self._load()
        instruction = _E5_QUERY_INSTRUCTION if _is_e5_model(self._settings.model_name) else ""
        vectors = model.encode(
            [f"{instruction}{cleaned}"],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self._settings.normalize,
            convert_to_numpy=True,
        )
        return [float(value) for value in vectors[0]]


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split())


def _is_e5_model(name: str) -> bool:
    lowered = (name or "").lower()
    return "e5" in lowered


def create_embedding_model(settings: ModelEmbeddingSettings) -> EmbeddingModel:
    """根据配置创建 Embedding 模型实例。"""
    model = _LocalEmbeddingModel(settings)
    # 移除强制加载，尊重懒加载机制。如果需要校验，应在调用方按需触发。
    return model


class EmbedderHealth(BaseModel):
    """Embedding 提供者的健康状态摘要。"""
    provider: str = "local"
    model_name: str
    model_loaded: bool = False
    dimension: int = 0


def describe_embedding_health(embedder: EmbeddingModel) -> EmbedderHealth:
    """返回安全的 Embedding 健康状态摘要。"""
    return EmbedderHealth(
        model_name=embedder.model_name,
        model_loaded=embedder.dimension > 0,
        dimension=embedder.dimension,
    )


def stable_hash(text: str, namespace: str = "chunk") -> str:
    """生成基于 SHA-256 的稳定标识符，用于去重。"""
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "EmbedderHealth",
    "EmbeddingModel",
    "create_embedding_model",
    "describe_embedding_health",
    "stable_hash",
]

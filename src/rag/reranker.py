"""重排序抽象与 Cross-Encoder 实现。"""

from __future__ import annotations

import threading
import typing
from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.rag.types import RetrievedDocument

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from sentence_transformers import CrossEncoder


class RerankerUnavailable(RuntimeError):
    """当配置的重排序器无法加载时抛出。"""


class RerankerOutputError(RerankerUnavailable):
    """当重排器输出与输入不一致时抛出。

    继承 RerankerUnavailable 是为了向后兼容既有捕获点，但语义上区分于
    "模型加载失败"——这表示模型正常返回了，但结果不可信。
    """


class RerankResult(BaseModel):
    """带有 cross-encoder 分数的重排文档。"""

    document: RetrievedDocument
    score: float


class Reranker(ABC):
    """使用 cross-encoder 分数对检索候选结果进行重排。"""

    name: str = "base"

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: typing.Sequence[RetrievedDocument],
        top_k: int = 4,
    ) -> list[RerankResult]:
        """返回按重排分数排序的头部文档。"""
        ...


class CrossEncoderReranker(Reranker):
    """本地 cross-encoder 重排器。

    启动时是可选的；只有当流水线实际尝试使用它且没有可用模型时，
    才会抛出 RerankerUnavailable。
    """

    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    def _load(self) -> CrossEncoder:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerUnavailable(
                "Cross-encoder 重排需要安装 sentence-transformers。"
            ) from exc

        with self._lock:
            if self._model is None:
                self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        documents: typing.Sequence[RetrievedDocument],
        top_k: int = 4,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []

        model = self._load()
        pairs = [(query, document.content) for document in documents]
        scores = model.predict(pairs, show_progress_bar=False)

        if len(scores) != len(documents):
            raise RerankerOutputError(
                f"Cross-encoder 返回了 {len(scores)} 个分数，但期望 {len(documents)} 个。"
            )

        decorated = [
            RerankResult(document=document, score=float(score))
            for document, score in zip(documents, scores, strict=True)
        ]
        decorated.sort(key=lambda item: item.score, reverse=True)
        return decorated[:top_k]


class IdentityReranker(Reranker):
    """无操作的重排器，保留向量的相似度排序。"""

    name = "identity"

    def rerank(
        self,
        query: str,
        documents: typing.Sequence[RetrievedDocument],
        top_k: int = 4,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        ordered = sorted(documents, key=lambda document: document.score, reverse=True)
        return [
            RerankResult(document=document, score=document.score)
            for document in ordered[:top_k]
        ]


# 注册表存工厂函数而非类，因为不同实现所需的构造参数不一致：
#   - CrossEncoderReranker 需要 model_name
#   - IdentityReranker 不需要任何参数
# 统一成 Callable[[str], Reranker] 后，工厂签名固定，Pyright 也能正确推断。
RerankerFactory = typing.Callable[[str], Reranker]

_REGISTRY: dict[str, RerankerFactory] = {}


def register_reranker(name: str, factory: RerankerFactory) -> None:
    """按名称注册一个重排器工厂。"""
    _REGISTRY[name.strip().lower()] = factory


def create_reranker(enabled: bool, model_name: str) -> Reranker:
    """创建配置的重排器，或直通回退。"""
    if not enabled:
        factory = _REGISTRY.get("identity")
        if factory is None:
            return IdentityReranker()
        return factory("")

    factory = _REGISTRY.get("cross_encoder")
    if factory is None:
        return CrossEncoderReranker(model_name)
    return factory(model_name)


# 两个内置工厂：签名统一为 (model_name: str) -> Reranker，
# IdentityReranker 忽略该参数。
register_reranker("cross_encoder", lambda model_name: CrossEncoderReranker(model_name))
register_reranker("identity", lambda _model_name: IdentityReranker())


__all__ = [
    "CrossEncoderReranker",
    "IdentityReranker",
    "RerankResult",
    "Reranker",
    "RerankerFactory",
    "RerankerOutputError",
    "RerankerUnavailable",
    "create_reranker",
    "register_reranker",
]

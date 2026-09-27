"""Reranking abstraction and cross-encoder implementation."""

from __future__ import annotations

import threading
import typing

from pydantic import BaseModel

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from sentence_transformers import CrossEncoder

    from src.rag.types import RetrievedDocument

_REGISTRY: dict[str, type[Reranker]] = {}


class RerankResult(BaseModel):
    """Reranked document with its cross-encoder score."""

    document: RetrievedDocument
    score: float


class Reranker:
    """Reorders retrieval candidates using a cross-encoder score."""

    name: str = "base"

    def rerank(self, query: str, documents: typing.Sequence[RetrievedDocument], top_k: int = 4) -> list[RerankResult]:
        """Return the top documents by rerank score."""

        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranker.

    It is optional at startup; :class:`RerankerUnavailable` is raised only when
    the pipeline actually attempts to use it without a usable model.
    """

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
                "Cross-encoder reranking requires sentence-transformers."
            ) from exc

        with self._lock:
            if self._model is None:
                self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model

    def rerank(self, query: str, documents: typing.Sequence[RetrievedDocument], top_k: int = 4) -> list[RerankResult]:
        if not documents:
            return []
        if top_k <= 0:
            return []

        model = self._load()
        pairs = [(query, document.content) for document in documents]
        scores = model.predict(pairs, show_progress_bar=False)
        decorated = [
            RerankResult(document=document, score=float(score))
            for document, score in zip(documents, scores, strict=False)
        ]
        decorated.sort(key=lambda item: item.score, reverse=True)
        return decorated[:top_k]


class IdentityReranker(Reranker):
    """No-op reranker that preserves vector similarity ordering."""

    def rerank(self, query: str, documents: typing.Sequence[RetrievedDocument], top_k: int = 4) -> list[RerankResult]:
        ordered = sorted(documents, key=lambda document: document.score, reverse=True)
        return [RerankResult(document=document, score=document.score) for document in ordered[:top_k]]


class RerankerUnavailable(RuntimeError):
    """Raised when the configured reranker cannot be loaded."""


def register_reranker(name: str, reranker_type: type[Reranker]) -> None:
    """Register a reranker implementation by name."""

    _REGISTRY[name.strip().lower()] = reranker_type


def create_reranker(enabled: bool, model_name: str) -> Reranker:
    """Create the configured reranker or a pass-through fallback."""

    if not enabled:
        return IdentityReranker()
    return CrossEncoderReranker(model_name)


register_reranker("cross_encoder", CrossEncoderReranker)
register_reranker("identity", IdentityReranker)


__all__ = [
    "CrossEncoderReranker",
    "IdentityReranker",
    "RerankResult",
    "Reranker",
    "RerankerUnavailable",
    "create_reranker",
    "register_reranker",
]

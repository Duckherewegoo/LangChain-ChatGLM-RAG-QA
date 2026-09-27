"""Embedding abstraction and default local implementation.

Embedding models are pluggable. Because changing the embedding model changes
vector geometry, :class:`EmbeddingModel` exposes a stable ``dimension`` so that
persistent indexes can reject mismatched collections before attempting search.
"""

from __future__ import annotations

import hashlib
import threading
import typing

from pydantic import BaseModel

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Sequence

    from sentence_transformers import SentenceTransformer

    from src.model.routing_types import ModelEmbeddingSettings

_E5_QUERY_INSTRUCTION = "query: "


class EmbeddingModel:
    """Embedding provider protocol."""

    name: str = "base"
    dimension: int = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document texts into embedding vectors."""

        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Encode a search query."""

        raise NotImplementedError


class _LocalEmbeddingModel(EmbeddingModel):
    """Sentence-Transformers-backed embedding model.

    It lazily loads the underlying model, normalizes empty or malformed input,
    and adds the E5 query instruction when the selected model expects it.
    """

    def __init__(self, settings: ModelEmbeddingSettings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _load(self) -> SentenceTransformer:
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
        cleaned = _normalize_text(text)
        if not cleaned:
            raise ValueError("Cannot embed an empty query.")

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
    """Create the configured local embedding model."""

    model = _LocalEmbeddingModel(settings)
    model._load()  # noqa: SLF001 - eager validation at construction time
    return model


class EmbedderHealth(BaseModel):
    """Embedding provider health summary."""

    provider: str = "local"
    model_name: str
    model_loaded: bool = False
    dimension: int = 0


def describe_embedding_health(embedder: EmbeddingModel) -> EmbedderHealth:
    """Return a safe health summary for an embedder."""

    model_name = getattr(getattr(embedder, "_settings", None), "model_name", "unknown")
    return EmbedderHealth(
        model_name=model_name,
        model_loaded=embedder.dimension > 0,
        dimension=embedder.dimension,
    )


def stable_hash(text: str, namespace: str = "chunk") -> str:
    """Create a stable SHA-256 derived identifier for deduplication."""

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

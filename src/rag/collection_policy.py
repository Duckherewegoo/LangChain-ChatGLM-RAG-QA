"""Vector collection policy and embedding dimension governance.

Different embedding models produce incompatible vector geometries. This module
assigns each model a persistent fingerprint, derives a unique collection name
from it, and refuses to open a collection whose stored dimension disagrees with
the active embedder. This prevents silent corruption when an operator swaps
the embedding model without rebuilding the index.
"""

from __future__ import annotations

import hashlib
import re
import typing

from pydantic import BaseModel

from src.exceptions import ConfigurationError, IngestError

if typing.TYPE_CHECKING:
    from src.settings import VectorStoreSettings

_FINGERPRINT_VERSION = 1
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_\-]")


class EmbeddingFingerprint(BaseModel):
    """Stable identity for an embedding configuration."""

    model_name: str
    dimension: int
    normalize: bool
    version: int = _FINGERPRINT_VERSION
    digest: str = ""

    @property
    def short_id(self) -> str:
        """Compact hash segment used in collection names."""

        if self.digest:
            return self.digest[:12]
        return self.compute_digest()[:12]

    def compute_digest(self) -> str:
        """Compute a canonical SHA-256 digest for the fingerprint."""

        payload = f"v{self.version}|{self.model_name}|{self.dimension}|{self.normalize}".encode()
        return hashlib.sha256(payload).hexdigest()

    def with_digest(self) -> EmbeddingFingerprint:
        """Return a copy whose digest is populated."""

        return EmbeddingFingerprint(
            model_name=self.model_name,
            dimension=self.dimension,
            normalize=self.normalize,
            version=self.version,
            digest=self.compute_digest(),
        )


class CollectionPolicy(BaseModel):
    """Rules mapping an embedding fingerprint to a physical collection."""

    base_name: str = "enterprise_knowledge"
    strategy: str = "model_fingerprint"
    max_name_length: int = 63

    def resolve(self, fingerprint: EmbeddingFingerprint) -> str:
        """Return the collection name for the active embedding model."""

        if self.strategy == "fixed":
            return self._safe(self.base_name)

        if self.strategy == "model_name":
            return self._safe(f"{self.base_name}_{self._slug(fingerprint.model_name)}")

        suffix = f"{fingerprint.dimension}d_{fingerprint.short_id}"
        return self._safe(f"{self.base_name}_{suffix}")

    def _safe(self, name: str) -> str:
        cleaned = _SAFE_NAME.sub("_", name).strip("_")
        if not cleaned:
            raise ConfigurationError("Resolved collection name is empty.")
        if len(cleaned) > self.max_name_length:
            cleaned = cleaned[-self.max_name_length :]
        return cleaned

    @staticmethod
    def _slug(text: str) -> str:
        return _SAFE_NAME.sub("_", text).strip("_").lower() or "unknown"


class CollectionCompatibilityError(IngestError):
    """Raised when an existing collection is incompatible with the active embedder."""

    status_code = 409
    error_code = "collection_incompatible"


def fingerprint_embedder(embedder: typing.Any, model_name: str) -> EmbeddingFingerprint:
    """Create a fingerprint from an embedder instance."""

    dimension = getattr(embedder, "dimension", 0)
    normalize = bool(getattr(embedder, "_settings", None) and getattr(embedder._settings, "normalize", True))
    if not dimension:
        raise ConfigurationError(
            "Embedder has no measurable dimension; load the model before opening a collection.",
            details={"model_name": model_name},
        )
    return EmbeddingFingerprint(
        model_name=model_name, dimension=dimension, normalize=normalize
    ).with_digest()


def create_collection_policy(settings: VectorStoreSettings) -> CollectionPolicy:
    """Create the policy declared in application settings."""

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
    """Return the collection name, raising if dimensions are incompatible."""

    collection = policy.resolve(fingerprint)
    if stored_dimension is None:
        return collection
    if stored_dimension != fingerprint.dimension:
        raise CollectionCompatibilityError(
            "Existing collection dimension does not match the active embedding model.",
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
    """Return a non-throwing compatibility summary for diagnostics."""

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

"""Plain data types for multi-model routing.

This module deliberately imports nothing from :mod:`src.settings` and nothing
from :mod:`src.model.manager`. It exists so that :mod:`src.settings` can
declare an ``AppSettings.model_router`` field of type
:class:`ModelRouterSettings` *without* triggering the import chain
``settings -> router -> manager -> settings``.

The separation is not arbitrary: Pydantic v2 resolves string-based forward
annotations against the module namespace when the class is created, and
:class:`pydantic_settings.BaseSettings` refuses to instantiate a class whose
``__pydantic_complete__`` flag is ``False``. The only way to keep that flag
``True`` while preserving a clean dependency direction is to place the routing
data classes in a leaf module.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ModelCapabilitySettings(BaseModel):
    """Capabilities required of a configured model."""

    supports_tools: bool = False
    supports_streaming: bool = True
    max_context_tokens: int | None = None


class ModelRoutingEntry(BaseModel):
    """One named model and its routing metadata.

    The conversion back to provider-agnostic settings is intentionally kept on
    this class so that the routing configuration stays decoupled from the
    concrete :class:`ModelChatSettings` definition.
    """

    name: str
    provider: str
    model_name: str
    base_url: str | None = None
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout_seconds: int = 60
    max_retries: int = 1
    capabilities: ModelCapabilitySettings = Field(default_factory=ModelCapabilitySettings)




class UseCaseBinding(BaseModel):
    """Map a business use case to an ordered list of model names."""

    use_case: str
    primary: str
    fallbacks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_chain(self) -> UseCaseBinding:
        chain = [self.primary, *self.fallbacks]
        if len(chain) != len(set(chain)):
            raise ValueError("Primary and fallback models must be distinct.")
        return self


class ModelRouterSettings(BaseModel):
    """Container for named models and use-case bindings."""

    models: dict[str, ModelRoutingEntry] = Field(default_factory=dict)
    use_cases: dict[str, UseCaseBinding] = Field(default_factory=dict)
    default_use_case: str = "chat"

    @model_validator(mode="after")
    def _validate_use_cases(self) -> ModelRouterSettings:
        for binding in self.use_cases.values():
            for reference in [binding.primary, *binding.fallbacks]:
                if reference not in self.models:
                    raise ValueError(
                        f"Use case {binding.use_case!r} references undefined model {reference!r}."
                    )
        if self.use_cases and self.default_use_case not in self.use_cases:
            raise ValueError(
                f"default_use_case {self.default_use_case!r} has no binding."
            )
        return self




class ModelEmbeddingSettings(BaseModel):
    """Embedding model parameters."""

    provider: str = Field(default="local", description="Embedding provider; reserved for future remote embeddings.")
    model_name: str = Field(default="BAAI/bge-small-zh-v1.5")
    device: str = Field(default="cpu")
    batch_size: int = Field(default=32, gt=0, le=512)
    normalize: bool = Field(default=True, description="L2-normalize embeddings for cosine similarity.")


__all__ = [
    "ModelCapabilitySettings",
    "ModelEmbeddingSettings",
    "ModelRouterSettings",
    "ModelRoutingEntry",
    "UseCaseBinding",
]

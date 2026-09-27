"""Multi-model configuration and routing.

This module resolves business use cases to configured models, constructs a
primary/secondary execution order, and decides whether a model may be used for
agent tool loops. It does not execute requests or call provider SDKs.

The plain configuration types (:class:`ModelCapabilitySettings`,
:class:`ModelRoutingEntry`, :class:`UseCaseBinding`, :class:`ModelRouterSettings`)
live in :mod:`src.model.routing_types` so that :mod:`src.settings` can declare
a ``model_router`` field without pulling in the full model layer.
"""

from __future__ import annotations

import typing

from src.exceptions import ConfigurationError, ResourceNotFoundError
from src.model.routing_types import (
    ModelCapabilitySettings,
    ModelRouterSettings,
    ModelRoutingEntry,
    UseCaseBinding,
)

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.model.fallback import FallbackExecutor
    from src.model.manager import ChatModel

from src.settings import ModelChatSettings


def to_provider_settings(entry: ModelRoutingEntry) -> ModelChatSettings:
    """Convert a routing entry into provider-agnostic settings.

    Kept as a module-level function rather than a method on
    :class:`ModelRoutingEntry` so that the routing data classes stay decoupled
    from the settings layer.
    """

    return ModelChatSettings(
        provider=entry.provider,
        model_name=entry.model_name,
        base_url=entry.base_url or "",
        api_key=entry.api_key,
        temperature=entry.temperature,
        max_tokens=entry.max_tokens,
        timeout_seconds=entry.timeout_seconds,
        max_retries=entry.max_retries,
    )


class ModelRoute:
    """Resolved execution plan for one use case."""

    def __init__(
        self,
        use_case: str,
        primary: ChatModel,
        primary_name: str,
        secondaries: list[tuple[str, ChatModel]],
    ) -> None:
        self.use_case = use_case
        self.primary = primary
        self.primary_name = primary_name
        self.secondaries = secondaries

    def ordered_names(self) -> list[str]:
        """Return model names in execution order."""

        return [self.primary_name, *(name for name, _ in self.secondaries)]

    def build_executor(self, *, max_attempts_per_provider: int = 1) -> FallbackExecutor:
        """Create a fallback executor using the resolved chain."""

        from src.model.fallback import FallbackExecutor

        return FallbackExecutor(
            self.primary,
            [model for _, model in self.secondaries],
            max_attempts_per_provider=max_attempts_per_provider,
        )


class ModelRouter:
    """Resolve use cases to model chains without calling providers."""

    def __init__(self, configuration: ModelRouterSettings) -> None:
        self._configuration = configuration
        self._cache: dict[str, ChatModel] = {}

    @property
    def configuration(self) -> ModelRouterSettings:
        return self._configuration

    def get_model(self, name: str) -> ChatModel:
        """Return a cached model instance by configured name."""

        if name in self._cache:
            return self._cache[name]

        entry = self._configuration.models.get(name)
        if entry is None:
            raise ResourceNotFoundError(
                f"Model {name!r} is not configured.",
                details={"available_models": sorted(self._configuration.models)},
            )

        _ensure_providers_registered()
        from src.model.manager import create_chat_model

        model = create_chat_model(to_provider_settings(entry))
        self._cache[name] = model
        return model

    def route(self, use_case: str | None = None) -> ModelRoute:
        """Resolve the model chain for a business use case."""

        binding = self._configuration.use_cases.get(
            use_case or self._configuration.default_use_case
        )
        if binding is None:
            raise ConfigurationError(
                f"No model binding for use case {use_case!r}.",
                details={"available_use_cases": sorted(self._configuration.use_cases)},
            )

        primary = self.get_model(binding.primary)
        secondaries: list[tuple[str, ChatModel]] = []
        for fallback_name in binding.fallbacks:
            secondaries.append((fallback_name, self.get_model(fallback_name)))
        return ModelRoute(binding.use_case, primary, binding.primary, secondaries)

    def supports_tools(self, name: str) -> bool:
        """Return True when the named model supports function calling."""

        entry = self._configuration.models.get(name)
        if entry is None:
            return False
        return entry.capabilities.supports_tools

    def has_use_case(self, use_case: str) -> bool:
        """Return True when the use case is configured."""

        return use_case in self._configuration.use_cases

    def clear_cache(self) -> None:
        """Discard cached model instances."""

        self._cache.clear()


def _ensure_providers_registered() -> None:
    """Import the provider package to trigger registration side effects.

    The providers subpackage registers implementations on import. Importing it
    lazily avoids forcing optional SDK availability at process startup.
    """

    import importlib

    importlib.import_module("src.model.providers")


def select_fallback_chain(
    preferred: str,
    available: typing.Iterable[str],
    *,
    exclude: typing.Container[str] = (),
) -> list[str]:
    """Order model names so the preferred model is first.

    This is a pure ordering helper used when a health report must override the
    static configuration order. It does not instantiate models.
    """

    names = [name for name in available if name not in exclude]
    ordered: list[str] = []
    if preferred in names:
        ordered.append(preferred)
    ordered.extend(name for name in names if name != preferred)
    return ordered


__all__ = [
    "ModelCapabilitySettings",
    "ModelRoute",
    "ModelRouter",
    "ModelRouterSettings",
    "ModelRoutingEntry",
    "UseCaseBinding",
    "select_fallback_chain",
]

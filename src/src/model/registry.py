"""Provider registry and discovery.

This module is the single authority for mapping provider identifiers to
factories. It deliberately contains no SDK imports, keeping provider
dependencies isolated and allowing registration to fail explicitly when an
optional dependency is absent.
"""

from __future__ import annotations

import typing

from src.exceptions import ConfigurationError

if typing.TYPE_CHECKING:
    from src.model.manager import ChatModel, ModelChatSettings, ProviderFactory

_REGISTRY: dict[str, ProviderFactory] = {}
_ALIASES: dict[str, str] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory under an exact identifier."""

    if not name or not isinstance(name, str):
        raise TypeError("Provider name must be a non-empty string.")
    if not callable(factory):
        raise TypeError("Provider factory must be callable.")

    _REGISTRY[name.strip().lower()] = factory


def register_alias(alias: str, canonical_name: str) -> None:
    """Map an alternate provider name to an already registered provider."""

    key = alias.strip().lower()
    target = canonical_name.strip().lower()
    if target not in _REGISTRY:
        raise ConfigurationError(
            f"Cannot alias {alias!r} to unregistered provider {canonical_name!r}.",
            details={"alias": alias, "canonical_name": canonical_name},
        )
    _ALIASES[key] = target


def resolve_factory(provider: str) -> ProviderFactory:
    """Return the factory for ``provider`` or a descriptive alias."""

    if not isinstance(provider, str) or not provider.strip():
        raise ConfigurationError(
            "Model provider must be a non-empty string.",
            details={"provider": provider},
        )

    key = provider.strip().lower()
    canonical = _ALIASES.get(key, key)
    factory = _REGISTRY.get(canonical)
    if factory is None:
        raise ConfigurationError(
            f"Unsupported model provider: {provider!r}.",
            details={"available_providers": sorted(_REGISTRY), "aliases": dict(_ALIASES)},
        )
    return factory


def create(provider: str, settings: ModelChatSettings) -> ChatModel:
    """Create a model instance using an explicitly named provider."""

    return resolve_factory(provider)(settings)


def available_providers() -> list[str]:
    """Return the registered provider identifiers."""

    return sorted(_REGISTRY)


def available_aliases() -> dict[str, str]:
    """Return current provider aliases."""

    return dict(_ALIASES)


def clear_registry() -> None:
    """Remove all providers and aliases.

    Intended for testing. Production code should not call this function.
    """

    _REGISTRY.clear()
    _ALIASES.clear()


def is_registered(name: str) -> bool:
    """Return True if the exact provider name is registered."""

    return name.strip().lower() in _REGISTRY


__all__ = [
    "available_aliases",
    "available_providers",
    "clear_registry",
    "create",
    "is_registered",
    "register_alias",
    "register_provider",
    "resolve_factory",
]

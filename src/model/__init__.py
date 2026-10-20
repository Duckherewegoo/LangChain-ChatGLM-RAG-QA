"""对话与嵌入模型的抽象层、提供者、路由与回退机制。

本模块作为模型层的统一外观（Facade），按需导出内部子模块的公共接口。
"""

import typing

if typing.TYPE_CHECKING:
    from model.embedding import (
        EmbedderHealth,
        EmbeddingModel,
        create_embedding_model,
        describe_embedding_health,
        stable_hash,
    )
    from model.errors import (
        ClassifiedError,
        ModelAPIConnectionError,
        ModelAuthenticationError,
        ModelConfigurationError,
        ModelContentPolicyError,
        ModelPayloadError,
        ModelRateLimitError,
        ModelTimeoutError,
        classify_model_error,
        is_retryable,
    )
    from model.fallback import FallbackAttempt, FallbackExecutor, FallbackResult
    from model.health import (
        HealthReport,
        ProviderHealth,
        check_chain,
        check_provider,
        select_provider,
    )
    from model.manager import (
        ProviderFactory,
        create_chat_model,
    )
    from model.protocol import (
        ChatModel,
        ChatRequest,
        ChatResponse,
    )
    from model.registry import (
        available_providers,
        clear_registry,
        is_registered,
        register_alias,
        register_provider,
        resolve_factory,
    )
    from model.router import (
        ModelRoute,
        ModelRouter,
        select_fallback_chain,
    )
    from model.routing_types import (
        ModelCapabilitySettings,
        ModelRouterSettings,
        ModelRoutingEntry,
        UseCaseBinding,
    )


def __getattr__(name: str) -> typing.Any:
    """PEP 562 惰性加载实现。
    
    防止在仅仅导入 model 时触发整个模型层及重型依赖的全量加载，
    同时避免潜在的循环导入死锁。
    """
    if name in ("ChatModel", "ChatRequest", "ChatResponse"):
        from model.protocol import ChatModel, ChatRequest, ChatResponse
        return locals()[name]

    if name in ("ProviderFactory", "create_chat_model"):
        from model.manager import ProviderFactory, create_chat_model
        return locals()[name]

    if name in ("EmbedderHealth", "EmbeddingModel", "create_embedding_model", "describe_embedding_health", "stable_hash"):
        from model.embedding import (
            EmbedderHealth,
            EmbeddingModel,
            create_embedding_model,
            describe_embedding_health,
            stable_hash,
        )
        return locals()[name]

    if name in ("ClassifiedError", "ModelAPIConnectionError", "ModelAuthenticationError", "ModelConfigurationError", "ModelContentPolicyError", "ModelPayloadError", "ModelRateLimitError", "ModelTimeoutError", "classify_model_error", "is_retryable"):
        from model.errors import (
            ClassifiedError,
            ModelAPIConnectionError,
            ModelAuthenticationError,
            ModelConfigurationError,
            ModelContentPolicyError,
            ModelPayloadError,
            ModelRateLimitError,
            ModelTimeoutError,
            classify_model_error,
            is_retryable,
        )
        return locals()[name]

    if name in ("FallbackAttempt", "FallbackExecutor", "FallbackResult"):
        from model.fallback import FallbackAttempt, FallbackExecutor, FallbackResult
        return locals()[name]

    if name in ("HealthReport", "ProviderHealth", "check_chain", "check_provider", "select_provider"):
        from model.health import HealthReport, ProviderHealth, check_chain, check_provider, select_provider
        return locals()[name]

    if name in ("available_providers", "clear_registry", "is_registered", "register_alias", "register_provider", "resolve_factory"):
        from model.registry import (
            available_providers,
            clear_registry,
            is_registered,
            register_alias,
            register_provider,
            resolve_factory,
        )
        return locals()[name]

    if name in ("ModelRoute", "ModelRouter", "select_fallback_chain"):
        from model.router import ModelRoute, ModelRouter, select_fallback_chain
        return locals()[name]

    if name in ("ModelCapabilitySettings", "ModelRouterSettings", "ModelRoutingEntry", "UseCaseBinding"):
        from model.routing_types import ModelCapabilitySettings, ModelRouterSettings, ModelRoutingEntry, UseCaseBinding
        return locals()[name]

    raise AttributeError(f"模块 {__name__!r} 没有属性 {name!r}")


__all__ = [
    "ChatModel",
    "ChatRequest",
    "ChatResponse",
    "ClassifiedError",
    "EmbedderHealth",
    "EmbeddingModel",
    "FallbackAttempt",
    "FallbackExecutor",
    "FallbackResult",
    "HealthReport",
    "ModelAPIConnectionError",
    "ModelAuthenticationError",
    "ModelCapabilitySettings",
    "ModelConfigurationError",
    "ModelContentPolicyError",
    "ModelPayloadError",
    "ModelRateLimitError",
    "ModelRoute",
    "ModelRouter",
    "ModelRouterSettings",
    "ModelRoutingEntry",
    "ModelTimeoutError",
    "ProviderFactory",
    "ProviderHealth",
    "UseCaseBinding",
    "available_providers",
    "check_chain",
    "check_provider",
    "classify_model_error",
    "clear_registry",
    "create_chat_model",
    "create_embedding_model",
    "describe_embedding_health",
    "is_registered",
    "is_retryable",
    "register_alias",
    "register_provider",
    "resolve_factory",
    "select_fallback_chain",
    "select_provider",
    "stable_hash",
]

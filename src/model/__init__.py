"""Chat and embedding model abstractions, providers, routing, and fallback."""

from src.model.embedding import (
    EmbedderHealth,
    EmbeddingModel,
    create_embedding_model,
    describe_embedding_health,
    stable_hash,
)
from src.model.errors import (
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
from src.model.fallback import FallbackAttempt, FallbackExecutor, FallbackResult
from src.model.health import (
    HealthReport,
    ProviderHealth,
    check_chain,
    check_provider,
    select_provider,
)
from src.model.manager import (
    ChatModel,
    ChatRequest,
    ChatResponse,
    ProviderFactory,
    create_chat_model,
)
from src.model.registry import (
    available_providers,
    clear_registry,
    is_registered,
    register_alias,
    register_provider,
    resolve_factory,
)
from src.model.router import (
    ModelRoute,
    ModelRouter,
    select_fallback_chain,
)
from src.model.routing_types import (
    ModelCapabilitySettings,
    ModelRouterSettings,
    ModelRoutingEntry,
    UseCaseBinding,
)

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

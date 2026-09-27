"""Application dependency container.

This module wires configuration, model routing, embedding, vector storage,
retrieval, agent tools, and the API layer once per process. Components are
created lazily and released through :func:`shutdown_container`.

The container never instantiates provider SDKs directly. It asks
:class:`ModelRouter` for a configured chain and wraps the result in a
:class:`FallbackExecutor` so that the rest of the application remains
agnostic of which model is currently active.
"""

from __future__ import annotations

import typing

from src.data_process.pipeline import IngestPipeline
from src.exceptions import ConfigurationError
from src.model import (
    ChatModel,
    EmbeddingModel,
    FallbackExecutor,
    ModelRouter,
    create_embedding_model,
)
from src.observability import get_logger
from src.rag.agent import AgentRequest as AgentRuntimeRequest
from src.rag.agent import AgentResponse, AgentRunner, create_agent_runner
from src.rag.collection_policy import (
    CollectionCompatibilityError,
    create_collection_policy,
    fingerprint_embedder,
)
from src.rag.orchestrator import AgentOrchestrator
from src.rag.retrieval import RetrievalPipeline
from src.rag.reviewer import AnswerReviewer
from src.rag.service import RAGService
from src.rag.strategies import (
    HybridRetrieval,
    QueryRewriter,
    RetrievalStrategy,
    VectorRetrievalStrategy,
    create_fuser,
    create_query_rewriter,
)
from src.rag.tools import ToolRegistry, create_rag_tool_registry
from src.rag.vector_store import VectorStore, create_vector_store

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.rag.agent import AgentResponse
    from src.settings import AppSettings

logger = get_logger(__name__)

_container: Container | None = None


class AgentTrailStore:
    """In-memory repository of recent agent audit trails.

    Trails are retained so callers can replay the exact sequence of reasoning,
    tool, and model events. This is intentionally an in-process cache; a
    production deployment should replace it with durable storage if audits must
    survive restarts.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max(1, max_entries)
        self._store: dict[str, typing.Any] = {}

    def put(self, request_id: str, trail: typing.Any) -> None:
        self._store[request_id] = trail
        while len(self._store) > self._max_entries:
            oldest = next(iter(self._store))
            self._store.pop(oldest, None)

    def get(self, request_id: str) -> typing.Any | None:
        return self._store.get(request_id)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._store


class Container:
    """Process-wide dependency container."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._router: ModelRouter | None = None
        self._executor: FallbackExecutor | None = None
        self._embedder: EmbeddingModel | None = None
        self._fingerprint: typing.Any = None
        self._vector_store: VectorStore | None = None
        self._retrieval: RetrievalPipeline | None = None
        self._rag_service: RAGService | None = None
        self._tool_registry: ToolRegistry | None = None
        self._agent_runner: AgentRunner | None = None
        self._ingest_pipeline: IngestPipeline | None = None
        self._query_rewriter: QueryRewriter | None = None
        self._strategies: list[tuple[str, RetrievalStrategy]] | None = None
        self._keyword_index: InMemoryBM25Index | None = None  # noqa: F821
        self._hybrid: HybridRetrieval | None = None
        self._reviewer: AnswerReviewer | None = None
        self._trail_store = AgentTrailStore()

    @property
    def settings(self) -> AppSettings:
        return self._settings

    @property
    def trail_store(self) -> AgentTrailStore:
        return self._trail_store

    def model_router(self) -> ModelRouter:
        """Return the model router, building it if necessary."""

        if self._router is None:
            self._router = self._build_router()
        return self._router

    def model_executor(self) -> FallbackExecutor:
        """Return the fallback executor for the default chat use case."""

        if self._executor is None:
            self._executor = self._build_executor()
        return self._executor

    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = create_embedding_model(self._settings.model_embedding)
        return self._embedder

    def embedding_fingerprint(self) -> typing.Any:
        """Return the stable fingerprint for the active embedder."""

        if self._fingerprint is None:
            self._fingerprint = fingerprint_embedder(
                self.embedder(), self._settings.model_embedding.model_name
            )
        return self._fingerprint

    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = self._build_vector_store()
        return self._vector_store

    def retrieval(self) -> RetrievalPipeline:
        if self._retrieval is None:
            self._retrieval = RetrievalPipeline(
                vector_store=self.vector_store(),
                embedder=self.embedder(),
                settings=self._settings.retrieval,
            )
        return self._retrieval

    def rag_service(self) -> RAGService:
        if self._rag_service is None:
            self._rag_service = RAGService(
                retrieval=self.retrieval(),
                chat_model=self._primary_model(),
                executor=self.model_executor(),
                embedder=self.embedder(),
                settings=self._settings.retrieval,
                guardrails=self._settings.guardrails,
            )
        return self._rag_service

    def tool_registry(self) -> ToolRegistry:
        if self._tool_registry is None:
            self._tool_registry = create_rag_tool_registry(self.rag_service())
        return self._tool_registry

    def agent_runner(self) -> AgentRunner:
        if self._agent_runner is None:
            self._agent_runner = create_agent_runner(
                executor=self.model_executor(),
                rag_service=self.rag_service(),
                tool_registry=self.tool_registry(),
                agent_settings=self._settings.agent,
                guardrails_settings=self._settings.guardrails,
            )
            self._wrap_agent_runner(self._agent_runner)
        return self._agent_runner

    def agent_orchestrator(self) -> AgentOrchestrator:
        """Return the multi-role orchestrator for the agent endpoint."""

        if not hasattr(self, "_orchestrator") or self._orchestrator is None:
            self._orchestrator: AgentOrchestrator | None = None
        if self._orchestrator is None:
            self._orchestrator = AgentOrchestrator(
                executor=self.model_executor(),
                retrieval=self.retrieval(),
                reviewer=self.reviewer(),
                registry=self.tool_registry(),
                agent_settings=self._settings.agent,
                guardrails_settings=self._settings.guardrails,
            )
        return self._orchestrator

    def query_rewriter(self) -> QueryRewriter:
        if self._query_rewriter is None:
            rewrite_cfg = getattr(self._settings.retrieval, "query_rewrite", None)
            use_llm = bool(getattr(rewrite_cfg, "method", "") == "llm") if rewrite_cfg else False
            model = self.model_router().route("chat").primary if use_llm else None
            self._query_rewriter = create_query_rewriter(rewrite_cfg, model=model)
        return self._query_rewriter

    def retrieval_strategies(self) -> list[tuple[str, RetrievalStrategy]]:
        if self._strategies is None:
            vector = VectorRetrievalStrategy(self.retrieval())
            strategies: list[tuple[str, RetrievalStrategy]] = [("vector", vector)]
            keyword = self._build_keyword_strategy()
            if keyword is not None:
                strategies.append(("keyword", keyword))
            self._strategies = strategies
        return self._strategies

    def _build_keyword_strategy(self) -> RetrievalStrategy | None:
        from src.rag.strategies import InMemoryBM25Index, KeywordRetrievalStrategy

        cfg = getattr(self._settings.retrieval, "keyword", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return None
        # 关键词索引在容器启动时为空；运行时由文档摄入管线填充
        self._keyword_index = InMemoryBM25Index()
        return KeywordRetrievalStrategy(self._keyword_index)

    def hybrid_retrieval(self) -> HybridRetrieval:
        if self._hybrid is None:
            self._hybrid = HybridRetrieval(
                rewriter=self.query_rewriter(),
                strategies=self.retrieval_strategies(),
                fuser=create_fuser(
                    getattr(self._settings.retrieval, "fusion", None)
                ),
                per_query_top_k=getattr(self._settings.retrieval, "per_query_top_k", 20),
            )
        return self._hybrid

    def reviewer(self) -> AnswerReviewer:
        if self._reviewer is None:
            review_cfg = getattr(self._settings.agent, "review", None)
            use_llm = bool(getattr(review_cfg, "use_llm", False)) if review_cfg else False
            model = self.model_router().route("chat").primary if use_llm else None
            self._reviewer = AnswerReviewer(
                model=model,
                max_evidence_chars=getattr(review_cfg, "max_evidence_chars", 6000) if review_cfg else 6000,
            )
        return self._reviewer

    def ingest_pipeline(self) -> IngestPipeline:
        if self._ingest_pipeline is None:
            from src.data_process.pipeline import DocumentChunker, DocumentLoader

            loader = DocumentLoader(self._settings.ingest.supported_extensions)
            chunker = DocumentChunker(
                chunk_size=self._settings.ingest.chunk_size,
                chunk_overlap=self._settings.ingest.chunk_overlap,
                min_characters=self._settings.ingest.min_characters,
            )
            self._ingest_pipeline = IngestPipeline(
                loader=loader, chunker=chunker, vector_store=self.vector_store()
            )
        return self._ingest_pipeline

    def shutdown(self) -> None:
        """Release shared resources."""

        for component in (
            self._agent_runner,
            self._reviewer,
            self._rag_service,
            self._hybrid,
            self._retrieval,
            self._tool_registry,
            self._vector_store,
        ):
            try:
                if hasattr(component, "shutdown"):
                    component.shutdown()
            except Exception as exc:
                logger.warning("component_shutdown_failed", extra={"error": str(exc)})

        self._vector_store = None
        self._router = None
        self._executor = None
        self._embedder = None
        self._fingerprint = None
        self._retrieval = None
        self._rag_service = None
        self._tool_registry = None
        self._agent_runner = None
        self._query_rewriter = None
        self._strategies = None
        self._keyword_index = None
        self._hybrid = None
        self._reviewer = None
        self._ingest_pipeline = None

    def _build_router(self) -> ModelRouter:
        configured = self._settings.model_router
        if configured and configured.models:
            return ModelRouter(configured)
        logger.info(
            "model_router_using_legacy_configuration",
            provider=self._settings.model_chat.provider,
            model_name=self._settings.model_chat.model_name,
        )
        return ModelRouter(_legacy_router_settings(self._settings))

    def _build_executor(self) -> FallbackExecutor:
        route = self.model_router().route("chat")
        return FallbackExecutor(
            primary=route.primary,
            secondaries=[model for _, model in route.secondaries],
            max_attempts_per_provider=max(1, self._settings.model_chat.max_retries),
        )

    def _primary_model(self) -> ChatModel:
        return self.model_router().route("chat").primary

    def _build_vector_store(self) -> VectorStore:
        policy = create_collection_policy(self._settings.retrieval.vector_store)
        fingerprint = self.embedding_fingerprint()
        collection = policy.resolve(fingerprint)

        store = create_vector_store(
            self._settings.retrieval.vector_store.type,
            collection=collection,
            persist_directory=self._settings.retrieval.vector_store.persist_directory,
        )
        stored_dimension = store.dimension()
        try:
            from src.rag.collection_policy import validate_compatibility

            validate_compatibility(policy, fingerprint, stored_dimension)
        except CollectionCompatibilityError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"Vector store compatibility check failed: {exc}",
                details={"collection": collection},
            ) from exc

        logger.info(
            "vector_store_initialized",
            collection=collection,
            strategy=policy.strategy,
            dimension=fingerprint.dimension,
        )
        return store

    def _wrap_agent_runner(self, runner: AgentRunner) -> None:
        """Record audit trails and preserve container semantics on failure."""

        original_arun = runner.arun

        async def arun(request: AgentRuntimeRequest) -> AgentResponse:
            response = await original_arun(request)
            self._trail_store.put(request.request_id, response.trail)
            return response

        runner.arun = arun  # type: ignore[assignment]
        original_run = runner.run

        def run(request: AgentRuntimeRequest) -> AgentResponse:  # type: ignore[assignment]
            response = original_run(request)
            self._trail_store.put(request.request_id, response.trail)
            return response

        runner.run = run  # type: ignore[assignment]


def _legacy_router_settings(settings: AppSettings):
    """Build a single-model router configuration from legacy chat settings."""

    from src.model.router import (
        ModelCapabilitySettings,
        ModelRouterSettings,
        ModelRoutingEntry,
        UseCaseBinding,
    )

    return ModelRouterSettings(
        models={
            "primary": ModelRoutingEntry(
                name="primary",
                provider=settings.model_chat.provider,
                model_name=settings.model_chat.model_name,
                base_url=settings.model_chat.base_url,
                api_key=settings.model_chat.api_key,
                temperature=settings.model_chat.temperature,
                max_tokens=settings.model_chat.max_tokens,
                timeout_seconds=settings.model_chat.timeout_seconds,
                max_retries=settings.model_chat.max_retries,
                capabilities=ModelCapabilitySettings(
                    supports_tools=True,
                    supports_streaming=True,
                ),
            )
        },
        use_cases={
            "chat": UseCaseBinding(use_case="chat", primary="primary"),
            "agent": UseCaseBinding(use_case="agent", primary="primary"),
        },
        default_use_case="chat",
    )


def initialize_container(settings: AppSettings | None = None) -> Container:
    """Create or replace the process-wide container."""

    global _container  # noqa: PLW0603 - intentional process-wide state

    from src.settings import load_settings

    if settings is None:
        settings = load_settings()
    if _container is not None:
        _container.shutdown()
    _container = Container(settings)
    logger.info("container_initialized", extra={})
    return _container


def get_container(reload: bool = False) -> Container:
    """Return the active container, creating it on first access."""

    global _container  # noqa: PLW0603 - intentional process-wide state

    if _container is None or reload:
        _container = initialize_container()
    return _container


def shutdown_container() -> None:
    """Shut down the active container if present."""

    global _container  # noqa: PLW0603 - intentional process-wide state

    if _container is not None:
        _container.shutdown()
        _container = None


__all__ = [
    "AgentTrailStore",
    "Container",
    "get_container",
    "initialize_container",
    "shutdown_container",
]

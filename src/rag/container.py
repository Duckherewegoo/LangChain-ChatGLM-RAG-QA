"""应用依赖容器。

本模块在每个进程内组装一次配置、模型路由、嵌入、向量存储、检索、智能体工具和 API 层。
组件按需（懒加载）创建，并通过 shutdown_container 释放。

容器绝不直接实例化提供者 SDK。它向 ModelRouter 请求已配置的模型链，并将结果包裹在
FallbackExecutor 中，从而让应用的其余部分对当前激活的模型保持无感知。
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
from src.rag.agent import AgentRunner, create_agent_runner
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
from src.rag.tools import Guardrails, ToolRegistry, create_rag_tool_registry
from src.rag.vector_store import VectorStore, create_vector_store

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from src.rag.strategies import InMemoryBM25Index
    from src.settings import AppSettings

logger = get_logger(__name__)

_container: "Container | None" = None


class AgentTrailStore:
    """最近智能体审计轨迹的内存仓库。

    保留轨迹是为了让调用方能够重放推理、工具和模型事件的精确序列。
    这刻意设计为进程内缓存；如果审计需要在重启后存活，生产部署应将其替换为持久化存储。
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
    """进程级依赖容器。"""

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
        self._keyword_index: InMemoryBM25Index | None = None
        self._hybrid: HybridRetrieval | None = None
        self._reviewer: AnswerReviewer | None = None
        self._orchestrator: AgentOrchestrator | None = None
        self._guardrails: Guardrails | None = None
        self._trail_store = AgentTrailStore()

    @property
    def settings(self) -> AppSettings:
        return self._settings

    @property
    def trail_store(self) -> AgentTrailStore:
        return self._trail_store

    # ---- 基础组件 ----------------------------------------------------------

    def guardrails(self) -> Guardrails:
        """返回全局 Guardrails 实例，供 API 层做输入校验。"""
        if self._guardrails is None:
            self._guardrails = Guardrails(
                self._settings.guardrails,
                agent_settings=self._settings.agent,
            )
        return self._guardrails

    def model_router(self) -> ModelRouter:
        """返回模型路由器，必要时进行构建。"""
        if self._router is None:
            self._router = self._build_router()
        return self._router

    def model_executor(self) -> FallbackExecutor:
        """返回默认聊天用例的回退执行器。"""
        if self._executor is None:
            self._executor = self._build_executor()
        return self._executor

    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = create_embedding_model(self._settings.model_embedding)
        return self._embedder

    def embedding_fingerprint(self) -> typing.Any:
        """返回当前 Embedder 的稳定指纹。"""
        if self._fingerprint is None:
            self._fingerprint = fingerprint_embedder(
                self.embedder(), self._settings.model_embedding.model_name
            )
        return self._fingerprint

    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = self._build_vector_store()
        return self._vector_store

    # ---- 检索层 ------------------------------------------------------------

    def retrieval(self) -> RetrievalPipeline:
        if self._retrieval is None:
            self._retrieval = RetrievalPipeline(
                vector_store=self.vector_store(),
                embedder=self.embedder(),
                settings=self._settings.retrieval,
            )
        return self._retrieval

    def query_rewriter(self) -> QueryRewriter:
        if self._query_rewriter is None:
            # settings.retrieval.query_rewrite 是字符串 "identity"/"expansion"/"llm"，
            # 而 create_query_rewriter 期望 dict，因此包一层。
            method = self._settings.retrieval.query_rewrite or "expansion"
            rewrite_cfg: dict[str, typing.Any] = {"method": method}
            use_llm = method == "llm"
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

        cfg = self._settings.retrieval.keyword
        if not cfg.enabled:
            return None
        self._keyword_index = InMemoryBM25Index()
        return KeywordRetrievalStrategy(self._keyword_index)

    def hybrid_retrieval(self) -> HybridRetrieval:
        if self._hybrid is None:
            # settings.retrieval.fusion 是 FusionSettings，create_fuser 期望 dict
            fusion_cfg = self._settings.retrieval.fusion.model_dump()
            self._hybrid = HybridRetrieval(
                rewriter=self.query_rewriter(),
                strategies=self.retrieval_strategies(),
                fuser=create_fuser(fusion_cfg),
                per_query_top_k=self._settings.retrieval.per_query_top_k,
            )
        return self._hybrid

    # ---- 服务层 ------------------------------------------------------------

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

    def reviewer(self) -> AnswerReviewer:
        if self._reviewer is None:
            review_cfg = self._settings.agent.review
            use_llm = review_cfg.use_llm
            model = self.model_router().route("chat").primary if use_llm else None
            self._reviewer = AnswerReviewer(
                model=model,
                max_evidence_chars=review_cfg.max_evidence_chars,
            )
        return self._reviewer

    def tool_registry(self) -> ToolRegistry:
        if self._tool_registry is None:
            self._tool_registry = create_rag_tool_registry(self.rag_service())
        return self._tool_registry

    # ---- Agent 层 ----------------------------------------------------------

    def agent_runner(self) -> AgentRunner:
        if self._agent_runner is None:
            self._agent_runner = create_agent_runner(
                executor=self.model_executor(),
                rag_service=self.rag_service(),
                tool_registry=self.tool_registry(),
                agent_settings=self._settings.agent,
                guardrails_settings=self._settings.guardrails,
            )
        return self._agent_runner

    def agent_orchestrator(self) -> AgentOrchestrator:
        """返回用于 Agent 端点的多角色编排器。"""
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

    # ---- 摄入层 ------------------------------------------------------------

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

    # ---- 生命周期 ----------------------------------------------------------

    def shutdown(self) -> None:
        """释放共享资源。

        优先调用 close()，没有则回退到 shutdown()，因为不同组件的清理方法名不统一。
        """
        for component in (
            self._agent_runner,
            self._reviewer,
            self._rag_service,
            self._hybrid,
            self._retrieval,
            self._tool_registry,
            self._vector_store,
        ):
            if component is None:
                continue
            try:
                close = getattr(component, "close", None)
                shutdown = getattr(component, "shutdown", None)
                if callable(close):
                    close()
                elif callable(shutdown):
                    shutdown()
            except Exception as exc:
                logger.warning(f"组件关闭失败 | error={exc!s}")

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
        self._orchestrator = None
        self._guardrails = None

    # ---- 内部构建 ----------------------------------------------------------

    def _build_router(self) -> ModelRouter:
        configured = self._settings.model_router
        if configured and configured.models:
            return ModelRouter(configured)
        logger.info(
            f"model_router_using_legacy_configuration | provider={self._settings.model_chat.provider} | model_name={self._settings.model_chat.model_name}"
        )
        return ModelRouter(_legacy_router_settings(self._settings))

    def _build_executor(self) -> FallbackExecutor:
        route = self.model_router().route("chat")
        # max_retries 语义是"重试次数"，max_attempts_per_provider 语义是"总尝试次数"
        max_attempts = max(1, self._settings.model_chat.max_retries + 1)
        return FallbackExecutor(
            primary=route.primary,
            secondaries=[model for _, model in route.secondaries],
            max_attempts_per_provider=max_attempts,
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
                f"向量库兼容性检查失败: {exc}",
                details={"collection": collection},
            ) from exc

        logger.info(
            f"vector_store_initialized | collection={collection} | strategy={policy.strategy} | dimension={fingerprint.dimension}"
        )
        return store


def _legacy_router_settings(settings: AppSettings) -> typing.Any:
    """从旧版聊天设置构建单模型路由器配置。"""
    from src.model.routing_types import (
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
    """创建或替换进程级容器。"""
    global _container

    from src.settings import load_settings

    if settings is None:
        settings = load_settings()
    if _container is not None:
        _container.shutdown()
    _container = Container(settings)
    logger.info("容器初始化完成。")
    return _container


def get_container(reload: bool = False) -> Container:
    """返回当前激活的容器，首次访问时创建。"""
    global _container

    if _container is None or reload:
        _container = initialize_container()
    return _container


def shutdown_container() -> None:
    """如果存在，关闭当前激活的容器。"""
    global _container

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

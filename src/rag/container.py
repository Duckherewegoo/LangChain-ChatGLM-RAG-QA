"""应用依赖容器。

本模块在每个进程内组装一次配置、模型路由、嵌入、向量存储、检索、智能体工具和 API 层。
组件按需（懒加载）创建，并通过 shutdown_container 释放。

容器绝不直接实例化提供者 SDK。它向 ModelRouter 请求已配置的模型链，并将结果包裹在
FallbackExecutor 中，从而让应用的其余部分对当前激活的模型保持无感知。

线程安全说明：
    get_container / initialize_container / shutdown_container 由模块级 RLock 保护，
    保证单例初始化的原子性。但锁不覆盖"请求使用容器"的生命周期——
    运行期调用 get_container(reload=True) 会关闭其他请求正在使用的容器。
    因此 reload 应仅用于启动期与测试。
"""

from __future__ import annotations

import threading
import typing

from data_process.pipeline import IngestPipeline
from exceptions import ConfigurationError
from model import (
    ChatModel,
    EmbeddingModel,
    FallbackExecutor,
    ModelRouter,
    create_embedding_model,
)
from observability import get_logger
from rag.agent import AgentRunner, create_agent_runner
from rag.collection_policy import (
    CollectionCompatibilityError,
    create_collection_policy,
    fingerprint_embedder,
)
from rag.orchestrator import AgentOrchestrator
from rag.retrieval import RetrievalPipeline
from rag.reviewer import AnswerReviewer
from rag.service import RAGService
from rag.strategies import (
    HybridRetrieval,
    QueryRewriter,
    RetrievalStrategy,
    VectorRetrievalStrategy,
    create_fuser,
    create_query_rewriter,
)
from rag.tools import Guardrails, ToolRegistry, create_rag_tool_registry
from rag.vector_store import VectorStore, create_vector_store

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from rag.agent_events import AgentAuditTrail
    from rag.strategies import InMemoryBM25Index
    from settings import AppSettings

logger = get_logger(__name__)

_container: "Container | None" = None
_container_lock = threading.RLock()


class AgentTrailStore:
    """最近智能体审计轨迹的内存仓库。

    保留轨迹是为了让调用方能够重放推理、工具和模型事件的精确序列。
    这刻意设计为进程内缓存；如果审计需要在重启后存活，生产部署应将其替换为持久化存储。

    并发说明：put 涉及"检查长度 + 淘汰 + 写入"三步，必须加锁；
    get / __contains__ 也加锁，保证与 put 的可观测一致性。
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max(1, max_entries)
        self._store: dict[str, "AgentAuditTrail"] = {}
        self._lock = threading.Lock()

    def put(self, request_id: str, trail: "AgentAuditTrail") -> None:
        with self._lock:
            self._store[request_id] = trail
            while len(self._store) > self._max_entries:
                oldest = next(iter(self._store))
                self._store.pop(oldest, None)

    def get(self, request_id: str) -> "AgentAuditTrail | None":
        with self._lock:
            return self._store.get(request_id)

    def __contains__(self, request_id: str) -> bool:
        with self._lock:
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
            # 注（M2）：rewrite 是便宜任务，理想应走 route("rewrite") 或小模型；
            # 当前无独立 use_case 配置，暂复用 "chat"。
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

    def _ensure_keyword_index(self) -> "InMemoryBM25Index | None":
        """确保关键词索引已构建，返回共享实例；未启用时返回 None。

        单一构造点：_build_keyword_strategy（查询侧）与 ingest_pipeline
        （写入/删除侧）都必须经此获取索引，否则会出现"写入 A、查询 B"
        的两份实例，P0-5 表现为"修了但没生效"。
        """
        if not self._settings.retrieval.keyword.enabled:
            return None
        if self._keyword_index is None:
            from rag.strategies import InMemoryBM25Index

            self._keyword_index = InMemoryBM25Index()
            # 说明（P0-5 已知边界）：内存索引不持久化，进程重启后为空；
            # 已有向量数据不会自动回填关键词索引。混合检索在重启后到下一次
            # ingest 之前，keyword 分支仍会返回空。长期方案是启动时从
            # vector_store 全量回填（BM25 需原文），或引入后台对账任务。
            logger.info("关键词索引已初始化（内存态，不持久化）")
        return self._keyword_index

    def _build_keyword_strategy(self) -> RetrievalStrategy | None:
        index = self._ensure_keyword_index()
        if index is None:
            return None
        from rag.strategies import KeywordRetrievalStrategy

        return KeywordRetrievalStrategy(index)

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
            # 已知取舍（P1-5）：审核模型不走 FallbackExecutor。primary 不可用
            # 时审核直接降级为规则审核（reviewer.py 的 except 分支），而规则
            # 审核无法检查越权披露等语义问题。生成链路有 fallback，审核链路
            # 没有——这是已知的不对称；修复需让 AnswerReviewer 接受 executor。
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
        """返回 LangGraph 智能体运行时。

        注意（P0-2）：本路径无审核节点——finalize 直接出答案。
        与 agent_orchestrator()（无 LangGraph、有审核）是两套独立实现，
        对外叙事需明确"两个端点、两种模式"。
        """
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
        """返回手写多角色编排器（含审核）。

        注意（P0-2）：本路径无 LangGraph 图，是 retrieve→reason→review
        手写循环。与 agent_runner()（有 LangGraph、无审核）是两套独立实现。
        """
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
            from data_process.pipeline import DocumentChunker, DocumentLoader

            loader = DocumentLoader(self._settings.ingest.supported_extensions)
            chunker = DocumentChunker(
                chunk_size=self._settings.ingest.chunk_size,
                chunk_overlap=self._settings.ingest.chunk_overlap,
                min_characters=self._settings.ingest.min_characters,
            )
            self._ingest_pipeline = IngestPipeline(
                loader=loader,
                chunker=chunker,
                vector_store=self.vector_store(),
                # 注入共享关键词索引：ingest 写入 / remove 删除会同步维护。
                # 未启用 keyword 检索时为 None，pipeline 内部会跳过同步。
                keyword_index=self._ensure_keyword_index(),
            )
        return self._ingest_pipeline

    # ---- 生命周期 ----------------------------------------------------------

    def __enter__(self) -> "Container":
        return self

    def __exit__(self, *exc_info: typing.Any) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """释放共享资源。

        优先调用 close()，没有则回退到 shutdown()，因为不同组件的清理方法名不统一。

        顺序按"高层 → 底层"：先关 agent_runner / orchestrator，再关 executor /
        embedder / router，避免底层被 close 后高层仍持有引用。
        """
        for component in (
            self._agent_runner,
            self._orchestrator,
            self._hybrid,
            self._reviewer,
            self._rag_service,
            self._retrieval,
            self._ingest_pipeline,
            self._tool_registry,
            self._vector_store,
            self._query_rewriter,
            self._executor,
            self._embedder,
            self._router,
            self._guardrails,
        ):
            if component is None:
                continue
            try:
                close_fn = getattr(component, "close", None)
                shutdown_fn = getattr(component, "shutdown", None)
                if callable(close_fn):
                    close_fn()
                elif callable(shutdown_fn):
                    shutdown_fn()
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
            "model_router_using_legacy_configuration | provider=%s | model_name=%s",
            self._settings.model_chat.provider,
            self._settings.model_chat.model_name,
        )
        return ModelRouter(_legacy_router_settings(self._settings))

    def _build_executor(self) -> FallbackExecutor:
        route = self.model_router().route("chat")
        # max_retries 是"每个 provider 的重试次数"（不含首次），
        # 因此每 provider 尝试次数 = max_retries + 1；
        # 整条链的总预算由 model_chat.max_total_attempts 单独控制，未设则不限。
        max_attempts = max(1, self._settings.model_chat.max_retries + 1)
        return FallbackExecutor(
            primary=route.primary,
            secondaries=[model for _, model in route.secondaries],
            max_attempts_per_provider=max_attempts,
            max_total_attempts=self._settings.model_chat.max_total_attempts,
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
        # 注意（M5）：空库首次初始化时 stored_dimension 可能为 None。
        # validate_compatibility 的 None 处理需在 collection_policy.py 确认；
        # 若其抛异常，首次部署空库会启动失败。
        try:
            from rag.collection_policy import validate_compatibility

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
    from model.routing_types import (
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
                    # 注意（M1）：此处假定旧版 chat 模型支持 tool calling。
                    # 若实际模型不支持，agent.py 的 registry.list_tools() 会在
                    # 模型侧报错。长期应改为从配置读取或启动时探测能力。
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

    from settings import load_settings

    if settings is None:
        settings = load_settings()
    with _container_lock:
        if _container is not None:
            _container.shutdown()
        _container = Container(settings)
    logger.info("容器初始化完成。")
    return _container


def get_container(reload: bool = False) -> Container:
    """返回当前激活的容器，首次访问时创建。

    reload=True 会关闭当前容器并创建新容器——这不是运行期安全操作：
    其他请求若已持有旧容器引用，会拿到已关闭的组件。请仅在启动期与测试使用。
    """
    global _container

    with _container_lock:
        if _container is None:
            return initialize_container()
        if reload:
            logger.warning(
                "容器 reload 请求：旧容器即将关闭，正在使用它的请求可能失败。"
                "reload 应仅用于启动期与测试。"
            )
            return initialize_container()
        return _container


def shutdown_container() -> None:
    """如果存在，关闭当前激活的容器。"""
    global _container

    with _container_lock:
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

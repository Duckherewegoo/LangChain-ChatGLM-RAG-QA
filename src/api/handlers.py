"""FastAPI 路由处理函数。

本模块承载所有请求处理逻辑，与 app 工厂解耦，保持 server.py 专注于装配。

职责边界：本模块只处理 orchestrator（retrieve → reason → review）路径。
LangGraph 智能体（agent_runner）的端点在 server.py 单独装配，不经本模块。
"""

from __future__ import annotations

import asyncio
import json
import re
import typing

from api.schemas import (
    AnswerRequestBody,
    AnswerResponseBody,
    CitationView,
    HealthModelView,
    HealthResponseBody,
    RetrievedDocumentView,
)
from exceptions import ConfigurationError
from model import ProviderHealth
from model.health import check_chain
from observability import get_logger
from rag.container import Container
from rag.agent import AgentRequest, AgentResponse
from rag.orchestrator import OrchestratorRequest
from rag.service import AnswerRequest

logger = get_logger(__name__)


def _ensure_vector_store(container: Container) -> None:
    """在启动时打开向量存储，让配置错误尽早暴露。"""
    _ = container.vector_store()


def _to_answer_request(body: AnswerRequestBody) -> AnswerRequest:
    return AnswerRequest(
        question=body.question.strip(),
        tenant_id=body.tenant_id,
        owner_id=body.owner_id,
        allowed_sources=body.allowed_sources,
        top_k=body.top_k,
        history=body.history,
        model=body.model,
        use_case=body.use_case,
    )


def _to_orchestrator_request(
    request: AnswerRequest | AgentRequest,
) -> OrchestratorRequest:
    """把 API 请求映射为 OrchestratorRequest。

    注意（M5）：OrchestratorRequest 不支持 top_k / model / use_case /
    additional_context —— 这些字段在 AnswerRequest / AgentRequest 上有，
    但编排器本身不消费。如需要请先扩展 OrchestratorRequest。

    注意（P1-2）：AnswerRequest 没有 request_id 字段；AgentRequest 有。
    有则透传，无则让 OrchestratorRequest 自行生成 uuid4。
    """
    request_id = getattr(request, "request_id", None)
    return OrchestratorRequest(
        question=request.question,
        tenant_id=request.tenant_id,
        owner_id=request.owner_id,
        allowed_sources=list(request.allowed_sources),
        **({"request_id": request_id} if request_id else {}),
    )


async def _run_via_orchestrator(
    container: Container,
    request: AnswerRequest | AgentRequest,
    additional_context: str | None,
) -> AgentResponse:
    """把请求提交给手写多角色编排器。

    编排器执行 retrieve → reason → review（审核阻断时按配置重试）。
    注意（P0-1）：本函数不使用 LangGraph —— 它只是 orchestrator 的适配器。
    LangGraph 运行时（agent_runner）在 server.py 单独装配，不经此函数。
    """
    orchestrator_request = _to_orchestrator_request(request)

    if additional_context:
        logger.warning(
            "additional_context 未在编排器中使用，已忽略 | request_id=%s",
            orchestrator_request.request_id,
        )

    return await _run_orchestrator(container, orchestrator_request)


async def _run_orchestrator(
    container: Container,
    request: OrchestratorRequest,
) -> AgentResponse:
    """执行编排器并把结果适配为 AgentResponse。

    编排器是同步阻塞的，必须通过 to_thread 提交到线程池，
    避免阻塞 FastAPI 的事件循环。
    """
    orchestrator = container.agent_orchestrator()

    logger.info(
        "编排请求开始 | request_id=%s | tenant=%s | question_len=%d",
        request.request_id,
        request.tenant_id,
        len(request.question),
    )

    result = await asyncio.to_thread(orchestrator.run, request)

    logger.info(
        "编排请求完成 | request_id=%s | status=%s | review_rounds=%d | duration=%.2fs",
        result.request_id,
        result.review_status,
        result.review_rounds,
        result.duration_seconds,
    )

    stopped_reason_map = {
        "approved": "review_approved",
        "rejected": "review_rejected",
        "warning": "completed_with_warning",
        "skipped": "completed_without_review",
        "not_applicable": "completed",
    }

    return AgentResponse(
        request_id=result.request_id,
        answer=result.answer,
        citations=result.citations,
        tool_calls=[],
        iterations=result.review_rounds,
        stopped_reason=stopped_reason_map.get(result.review_status, "completed"),
        model=result.model,
        documents=result.documents,
        # TODO（M4）：orchestrator.trail 已含 MODEL_CALL 事件；待
        # AgentAuditTrail 公开稳定的事件迭代 API 后，从这里聚合真实值。
        attempted_models=[],
        fallback_used=False,
        trail=result.trail,
        duration_seconds=result.duration_seconds,
    )


def _agent_to_answer(response: AgentResponse, original_query: str) -> AnswerResponseBody:
    """将智能体响应转换为 API 响应体。"""
    return AnswerResponseBody(
        answer=response.answer,
        query=original_query,
        citations=[_safe_citation(item) for item in response.citations],
        documents=[_safe_document(doc) for doc in response.documents],
        context_used=bool(response.documents),
        model=response.model,
        tokens={},
        attempted_models=response.attempted_models,
        fallback_used=response.fallback_used,
        request_id=response.request_id,
    )


def _safe_citation(item: dict[str, typing.Any]) -> CitationView:
    """容错构建 CitationView，忽略未知字段。

    白名单与 orchestrator._build_citations 的输出对齐：index / source /
    title / page / score。chunk_id 不在编排器输出中，去掉以免误导。
    """
    allowed = {"index", "source", "title", "page", "score"}
    return CitationView(**{k: v for k, v in item.items() if k in allowed})


def _safe_document(item: dict[str, typing.Any]) -> RetrievedDocumentView:
    """容错构建 RetrievedDocumentView，忽略未知字段。"""
    allowed = {
        "document_id", "chunk_id", "title", "source", "page",
        "score", "rerank_score", "snippet", "content",
    }
    return RetrievedDocumentView(**{k: v for k, v in item.items() if k in allowed})


def _build_health(container: Container, *, liveness: bool) -> HealthResponseBody:
    """构建 liveness / readiness 健康检查响应。

    liveness=True：只返回进程与配置信息，不触碰向量库或 embedder。
    K8s 存活探针不应依赖外部组件——向量库抖动会导致 Pod 重启循环。

    liveness=False：读取向量库与 embedder，检查维度兼容性。失败仅影响
    readiness，不影响存活。
    """
    settings = container.settings

    if liveness:
        return HealthResponseBody(
            status="ok",
            app_env=settings.app_env,
            api_configured=settings.model_chat.is_configured,
            vector_store=settings.retrieval.vector_store.type,
            collection="n/a",
            embedding_model=settings.model_embedding.model_name,
            embedding_dimension=0,
            collection_compatible=True,
            primary_model=None,
            available_models=[],
            degraded=False,
        )

    store = container.vector_store()
    fingerprint = container.embedding_fingerprint()
    _, collection_compatible = _collection_status(container)
    primary_name, _ = _primary_route(container.model_router())

    return HealthResponseBody(
        status="ok" if collection_compatible else "incompatible",
        app_env=settings.app_env,
        api_configured=settings.model_chat.is_configured,
        vector_store=settings.retrieval.vector_store.type,
        collection=store.describe().get(
            "collection", settings.retrieval.vector_store.collection
        ),
        embedding_model=settings.model_embedding.model_name,
        embedding_dimension=fingerprint.dimension,
        collection_compatible=collection_compatible,
        primary_model=primary_name,
        available_models=[],
        degraded=False,
    )


async def _build_model_health(container: Container) -> HealthResponseBody:
    """构建深度模型健康检查响应（会发起真实的模型 ping）。"""
    settings = container.settings
    router = container.model_router()

    primary_name, secondaries = _primary_route(router)

    if not primary_name:
        raise ConfigurationError(
            "未配置任何模型用例；请配置 model_router.use_cases。"
        )

    # P1-3：check_chain 走同步 HTTP ping，必须提交到线程池，
    # 否则会阻塞整个事件循环（K8s liveness 高频调用时尤为明显）。
    report = await asyncio.to_thread(
        check_chain,
        primary_provider=primary_name,
        primary=router.get_model(primary_name),
        secondaries=secondaries,
        tool_enabled=router.supports_tools(primary_name),
    )

    # P1-4：深度检查也读真实向量库状态，不再硬编码 collection_compatible=True。
    _, collection_compatible = _collection_status(container)
    collection_name = (
        container.vector_store().describe().get(
            "collection", settings.retrieval.vector_store.collection
        )
    )

    return HealthResponseBody(
        status="ok" if report.is_healthy() else "unhealthy",
        app_env=settings.app_env,
        api_configured=settings.model_chat.is_configured,
        vector_store=settings.retrieval.vector_store.type,
        collection=collection_name,
        embedding_model=settings.model_embedding.model_name,
        embedding_dimension=container.embedding_fingerprint().dimension,
        collection_compatible=collection_compatible,
        models=[_health_to_view(h) for h in report.providers],
        primary_model=report.primary,
        available_models=report.available,
        degraded=report.degraded,
    )


def _health_to_view(health: ProviderHealth) -> HealthModelView:
    return HealthModelView(
        provider=health.provider,
        model_name=health.model_name,
        configured=health.configured,
        reachable=health.reachable,
        latency_seconds=health.latency_seconds,
        error_type=health.error_type,
        error_message=_safe_health_message(health.error_message),
        supports_tools=health.supports_tools,
    )


# 收紧 base64 分支：长度 40+ 且必须含 padding，降低误伤常规 base64 内容
# （哈希、长 ID）的概率；真 secret 通常带 = 或更长。
_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,}|[A-Za-z0-9+/]{40,}={1,2})"
)


def _safe_health_message(message: str | None) -> str | None:
    """返回脱敏后的健康检查错误消息。"""
    if not message:
        return None
    redacted = _SECRET_PATTERN.sub("[REDACTED]", message)
    return redacted[:300]


def _sse(payload: dict[str, typing.Any]) -> str:
    """把 payload 序列化为一条 SSE 消息（用于 /v1/agent/stream）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _collection_status(container: Container) -> tuple[int | None, bool]:
    """返回 (实际存储维度, 是否与当前 embedder 指纹兼容)。

    stored_dim 为 None 表示空库或读取失败，此时视为兼容（首次初始化场景）。
    """
    store = container.vector_store()
    fingerprint = container.embedding_fingerprint()
    try:
        stored_dim = store.dimension()
    except Exception as exc:
        logger.error("读取向量存储维度失败 | error=%s", exc)
        stored_dim = None
    return stored_dim, stored_dim in (None, fingerprint.dimension)


def _primary_route(
    router: typing.Any,
) -> tuple[str, list[tuple[str, typing.Any]]]:
    """返回首个已配置用例的 (primary_name, secondaries)。

    遍历顺序 chat → agent，与历史行为一致。
    """
    for use_case in ("chat", "agent"):
        if router.has_use_case(use_case):
            route = router.route(use_case)
            return route.primary_name, list(route.secondaries)
    return "", []

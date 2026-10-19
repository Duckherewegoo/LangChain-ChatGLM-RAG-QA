"""FastAPI 路由处理函数。

本模块承载所有请求处理逻辑，与 app 工厂解耦，保持 server.py 专注于装配。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import typing

from src.api.schemas import (
    AnswerRequestBody,
    AnswerResponseBody,
    CitationView,
    HealthModelView,
    HealthResponseBody,
    RetrievedDocumentView,
)
from src.exceptions import ConfigurationError
from src.model import ProviderHealth
from src.model.health import check_chain
from src.rag.container import Container
from src.rag.agent import AgentRequest, AgentResponse
from src.rag.orchestrator import OrchestratorRequest
from src.rag.service import AnswerRequest

logger = logging.getLogger(__name__)


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


async def _run_agent(
    container: Container,
    request: AnswerRequest | AgentRequest,
    additional_context: str | None,
) -> AgentResponse:
    """将请求路由到多角色编排器。

    编排器执行 retrieve -> reason -> review（审核阻断时按配置重试），
    这是完整的多角色协作契约。
    """
    orchestrator_request = OrchestratorRequest(
        question=request.question,
        tenant_id=request.tenant_id,
        owner_id=request.owner_id,
        allowed_sources=list(request.allowed_sources),
        history=list(request.history),
    )

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
    """容错构建 CitationView，忽略未知字段。"""
    allowed = {"index", "source", "title", "page", "chunk_id"}
    return CitationView(**{k: v for k, v in item.items() if k in allowed})


def _safe_document(item: dict[str, typing.Any]) -> RetrievedDocumentView:
    """容错构建 RetrievedDocumentView，忽略未知字段。"""
    allowed = {
        "document_id", "chunk_id", "title", "source", "page",
        "score", "rerank_score", "snippet", "content",
    }
    return RetrievedDocumentView(**{k: v for k, v in item.items() if k in allowed})


def _build_health(container: Container, *, liveness: bool) -> HealthResponseBody:
    """构建 liveness / readiness 健康检查响应（不发起模型调用）。"""
    settings = container.settings
    store = container.vector_store()
    fingerprint = container.embedding_fingerprint()

    try:
        stored_dim = store.dimension()
    except Exception as exc:
        logger.error("读取向量存储维度失败 | error=%s", exc)
        stored_dim = None

    collection_compatible = stored_dim in (None, fingerprint.dimension)

    router = container.model_router()
    primary_name = ""
    for use_case in ("chat", "agent"):
        if router.has_use_case(use_case):
            primary_name = router.route(use_case).primary_name
            break

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

    primary_name = ""
    secondaries: list[tuple[str, typing.Any]] = []
    for use_case in ("chat", "agent"):
        if router.has_use_case(use_case):
            route = router.route(use_case)
            primary_name = route.primary_name
            secondaries = list(route.secondaries)
            break

    if not primary_name:
        raise ConfigurationError(
            "未配置任何模型用例；请配置 model_router.use_cases。"
        )

    report = check_chain(
        primary_provider=primary_name,
        primary=router.get_model(primary_name),
        secondaries=secondaries,
        tool_enabled=router.supports_tools(primary_name),
    )

    return HealthResponseBody(
        status="ok" if report.is_healthy() else "unhealthy",
        app_env=settings.app_env,
        api_configured=settings.model_chat.is_configured,
        vector_store=settings.retrieval.vector_store.type,
        collection="n/a",
        embedding_model=settings.model_embedding.model_name,
        embedding_dimension=container.embedding_fingerprint().dimension,
        collection_compatible=True,
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


_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,}|[A-Za-z0-9+/]{32,}={0,2})"
)


def _safe_health_message(message: str | None) -> str | None:
    """返回脱敏后的健康检查错误消息。"""
    if not message:
        return None
    redacted = _SECRET_PATTERN.sub("[REDACTED]", message)
    return redacted[:300]


def _sse(payload: dict[str, typing.Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

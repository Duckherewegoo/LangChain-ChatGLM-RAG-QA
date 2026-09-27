"""Route handler functions for the FastAPI application.

This module contains all request handler logic. It is separated from the app
factory to keep ``server.py`` focused on wiring rather than business logic.
"""

from __future__ import annotations

import json
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
from src.rag import Container
from src.rag.agent import AgentResponse, AgentRuntimeRequest
from src.rag.orchestrator import OrchestratorRequest
from src.rag.service import AnswerRequest


def _ensure_vector_store(container: Container) -> None:
    """Open the vector store so startup failures surface immediately."""

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
    container: Container, request: AnswerRequest | AgentRuntimeRequest, additional_context: str | None
) -> typing.Any:
    """Route an agent request through the multi-role orchestrator.

    The orchestrator runs retrieve -> reason -> review (with optional retry on
    blocking review issues), which is the full harness contract. The legacy
    single-graph runner is kept for backwards compatibility but is not the
    default path anymore.
    """
    if isinstance(request, AnswerRequest):
        question = request.question
        tenant_id = request.tenant_id
        owner_id = request.owner_id
        allowed_sources = request.allowed_sources
        history = request.history
    else:
        question = request.question
        tenant_id = request.tenant_id
        owner_id = request.owner_id
        allowed_sources = request.allowed_sources
        history = request.history

    orchestrator_request = OrchestratorRequest(
        question=question,
        tenant_id=tenant_id,
        owner_id=owner_id,
        allowed_sources=list(allowed_sources),
        history=list(history),
    )
    return await _run_orchestrator(container, orchestrator_request)


async def _run_orchestrator(
    container: Container, request: OrchestratorRequest
) -> typing.Any:
    """Execute the orchestrator and return an adapter shaped like AgentResponse."""

    orchestrator = container.agent_orchestrator()
    result = orchestrator.run(request)

    return AgentResponse(
        request_id=result.request_id,
        answer=result.answer,
        citations=result.citations,
        tool_calls=[],
        iterations=result.review_rounds,
        stopped_reason="review_approved" if result.review_status == "approved" else "completed",
        model=result.model,
        documents=result.documents,
        attempted_models=[],
        fallback_used=False,
        trail=result.trail,
    )


def _agent_to_answer(response: typing.Any) -> AnswerResponseBody:
    return AnswerResponseBody(
        answer=response.answer,
        query=response.request_id,
        citations=[CitationView(**item) for item in response.citations],
        documents=[RetrievedDocumentView(**document) for document in response.documents],
        context_used=bool(response.documents),
        model=response.model,
        tokens={"tool_calls": len(response.tool_calls)},
        attempted_models=response.attempted_models,
        fallback_used=response.fallback_used,
        request_id=response.request_id,
    )


def _build_health(container: Container, *, liveness: bool) -> HealthResponseBody:
    settings = container.settings
    store = container.vector_store()
    fingerprint = container.embedding_fingerprint()
    collection_compatible = store.dimension() in (None, fingerprint.dimension)
    primary_name = ""
    for use_case in ("chat", "agent"):
        if container.model_router().has_use_case(use_case):
            primary_name = container.model_router().route(use_case).primary_name
            break

    return HealthResponseBody(
        status="ok" if collection_compatible else "incompatible",
        app_env=settings.app_env,
        api_configured=settings.model_chat.is_configured,
        vector_store=settings.retrieval.vector_store.type,
        collection=store.describe().get("collection", settings.retrieval.vector_store.collection),
        embedding_model=settings.model_embedding.model_name,
        embedding_dimension=fingerprint.dimension,
        collection_compatible=collection_compatible,
        primary_model=primary_name,
        available_models=[],
        degraded=False,
    )


async def _build_model_health(container: Container) -> HealthResponseBody:
    settings = container.settings
    router = container.model_router()

    primary_name = ""
    secondaries: list[tuple[str, typing.Any]] = []
    for use_case in ("chat", "agent"):
        if router.has_use_case(use_case):
            route = router.route(use_case)
            primary_name = route.primary_name
            secondaries = [(name, model) for name, model in route.secondaries]
            break

    if not primary_name:
        raise ConfigurationError("No model use case is configured; configure model_router.use_cases.")

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
        models=[_health_to_view(health) for health in report.providers],
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


def _safe_health_message(message: str | None) -> str | None:
    if not message:
        return None
    lowered = message.lower()
    if any(secret in lowered for secret in ("api_key", "secret", "token")):
        return "Provider call failed; see service logs for details."
    return message[:300]


def _sse(payload: dict[str, typing.Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"




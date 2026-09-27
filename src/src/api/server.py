"""FastAPI application and HTTP routes.

This layer is intentionally thin: it validates transport contracts, attaches
observability context, and delegates business logic to the container. It does
not import provider SDKs or implement retrieval algorithms.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from src.api.handlers import (
    _agent_to_answer,
    _build_health,
    _build_model_health,
    _ensure_vector_store,
    _run_agent,
    _sse,
    _to_answer_request,
)
from src.api.middleware import map_exception
from src.api.schemas import (
    AgentRequestBody,
    AgentResponseBody,
    AgentToolCallView,
    AnswerRequestBody,
    AnswerResponseBody,
    CitationView,
    HealthResponseBody,
    IngestRequestBody,
    IngestResponseBody,
    MessageResponseBody,
    RetrievedDocumentView,
)
from src.exceptions import RAGError, ResourceNotFoundError, ValidationError
from src.observability import (
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    collect_metrics,
    configure_observability,
    finish_request,
    get_logger,
    request_id,
    start_request,
)
from src.rag import (
    Container,
    get_container,
)
from src.rag.agent import AgentRequest as AgentRuntimeRequest
from src.rag.agent_events import AgentEventType
from src.rag.service import AnswerRequest
from src.settings import load_settings

logger = get_logger(__name__)


def create_app(container: Container | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = load_settings()
    configure_observability(settings.log_level, settings.observability.metrics_enabled)
    dependencies = container or get_container()

    app = FastAPI(
        title="Enterprise RAG Assistant",
        version="0.1.0",
        description="LangChain/LangGraph RAG service with a pluggable ChatGLM/Qwen/DeepSeek model layer.",
    )
    app.state.container = dependencies

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id", "X-Process-Time"],
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        correlation_id = start_request(request.headers.get("X-Request-Id"))
        endpoint = f"{request.method} {request.url.path}"
        time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            status_code, envelope = map_exception(exc, request_id=correlation_id)
            response = JSONResponse(status_code=status_code, content=envelope.model_dump())
        finally:
            elapsed = finish_request()
        response.headers["X-Request-Id"] = correlation_id
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        if REQUEST_LATENCY:
            REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(elapsed)
        if REQUESTS_TOTAL:
            REQUESTS_TOTAL.labels(method=request.method, endpoint=endpoint, status_code=response.status_code).inc()
        return response

    @app.exception_handler(Exception)
    async def _fallback_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        status_code, envelope = map_exception(exc, request_id=request_id() or None)
        return JSONResponse(status_code=status_code, content=envelope.model_dump())

    @app.exception_handler(RAGError)
    async def _domain_exception_handler(request: Request, exc: RAGError) -> JSONResponse:
        status_code, envelope = map_exception(exc, request_id=request_id() or None)
        return JSONResponse(status_code=status_code, content=envelope.model_dump())

    @app.get("/healthz", response_model=HealthResponseBody)
    async def healthz() -> HealthResponseBody:
        _ensure_vector_store(dependencies)
        return _build_health(dependencies, liveness=True)

    @app.get("/readyz", response_model=HealthResponseBody)
    async def readyz() -> HealthResponseBody:
        _ensure_vector_store(dependencies)
        return _build_health(dependencies, liveness=False)

    @app.get("/healthz/models", response_model=HealthResponseBody)
    async def models_health() -> HealthResponseBody:
        return await _build_model_health(dependencies)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint() -> Response:
        payload = collect_metrics()
        if payload is None:
            return PlainTextResponse("# Prometheus client not installed\n", media_type="text/plain")
        return PlainTextResponse(payload.decode("utf-8"), media_type="text/plain; version=0.0.4")

    @app.post("/v1/answer", response_model=AnswerResponseBody)
    async def create_answer(body: AnswerRequestBody) -> AnswerResponseBody:
        dependencies.settings.guardrails.check_question(body.question)
        service = dependencies.rag_service()
        request = _to_answer_request(body)

        if body.use_agent:
            agent_response = await _run_agent(dependencies, request, body.additional_context)
            return _agent_to_answer(agent_response)

        result = await service.aanswer(request)
        return AnswerResponseBody(
            answer=result.answer,
            query=result.query,
            citations=[CitationView(**citation.model_dump()) for citation in result.citations],
            documents=[RetrievedDocumentView(**document) for document in result.context.documents],
            context_used=result.context_used,
            model=result.model,
            tokens=result.tokens,
            attempted_models=result.attempted_models,
            fallback_used=result.fallback_used,
            request_id=request_id() or None,
        )

    @app.post("/v1/agent", response_model=AgentResponseBody)
    async def run_agent(body: AgentRequestBody) -> AgentResponseBody:
        dependencies.settings.guardrails.check_question(body.question)
        runtime_request = AgentRuntimeRequest(
            question=body.question.strip(),
            tenant_id=body.tenant_id,
            owner_id=body.owner_id,
            allowed_sources=body.allowed_sources,
            history=body.history,
            additional_context=body.additional_context,
            use_case=body.use_case,
        )
        response = await _run_agent(dependencies, runtime_request, body.additional_context)
        return AgentResponseBody(
            request_id=response.request_id,
            answer=response.answer,
            citations=response.citations,
            tool_calls=[AgentToolCallView(**call.model_dump()) for call in response.tool_calls],
            iterations=response.iterations,
            stopped_reason=response.stopped_reason,
            model=response.model,
            documents=response.documents,
            attempted_models=response.attempted_models,
            fallback_used=response.fallback_used,
        )

    @app.post("/v1/agent/stream")
    async def stream_agent(body: AgentRequestBody):
        dependencies.settings.guardrails.check_question(body.question)
        runtime_request = AgentRuntimeRequest(
            question=body.question.strip(),
            tenant_id=body.tenant_id,
            owner_id=body.owner_id,
            allowed_sources=body.allowed_sources,
            history=body.history,
            additional_context=body.additional_context,
            use_case=body.use_case,
        )
        runner = dependencies.agent_runner()

        async def event_stream():
            yield _sse({"event": AgentEventType.STARTED, "request_id": runtime_request.request_id})
            try:
                async for event in runner.stream(runtime_request):
                    yield _sse(event)
            except ValidationError as exc:
                yield _sse({"event": "error", "error_code": exc.error_code, "message": exc.message})
            except Exception as exc:
                _code, envelope = map_exception(exc, request_id=runtime_request.request_id)
                yield _sse({"event": "error", **envelope.model_dump()})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/v1/agent/{request_id}/trail", response_model=dict)
    async def get_agent_trail(request_id_path: str) -> dict:
        store = dependencies.agent_runner().trail_store
        trail = store.get(request_id_path)
        if trail is None:
            raise ResourceNotFoundError(f"No agent trail found for request {request_id_path!r}.")
        return trail.to_jsonl()

    @app.post("/v1/ingest", response_model=IngestResponseBody)
    async def ingest(body: IngestRequestBody) -> IngestResponseBody:
        pipeline = dependencies.ingest_pipeline()
        stats = pipeline.ingest_path(
            body.path,
            tenant_id=body.tenant_id,
            owner_id=body.owner_id,
            title=body.title,
            tags=body.tags,
        )
        return IngestResponseBody(
            status="accepted",
            message=f"Ingested {stats.chunks_stored} chunks from {stats.scanned} files.",
            document_id=stats.document_id,
            scanned=stats.scanned,
            chunks_stored=stats.chunks_stored,
            bytes_size=stats.bytes_size,
            request_id=request_id() or None,
        )

    @app.delete("/v1/documents/{document_id}", response_model=MessageResponseBody)
    async def delete_document(document_id: str, tenant_id: str = "default") -> MessageResponseBody:
        pipeline = dependencies.ingest_pipeline()
        removed = pipeline.remove_document(document_id, tenant_id)
        return MessageResponseBody(
            status="deleted",
            message=f"Removed {removed} chunks for document {document_id}.",
            request_id=request_id() or None,
        )

    @app.get("/v1/retrieve", response_model=AnswerResponseBody)
    async def retrieve(
        question: str,
        tenant_id: str = "default",
        top_k: int = 8,
        owner_id: str | None = None,
    ) -> AnswerResponseBody:
        dependencies.settings.guardrails.check_question(question)
        service = dependencies.rag_service()
        result = service.retrieve_only(
            AnswerRequest(question=question, tenant_id=tenant_id, owner_id=owner_id, top_k=top_k)
        )
        return AnswerResponseBody(
            answer="",
            query=question,
            documents=[RetrievedDocumentView(**document.model_dump()) for document in result.documents],
            context_used=not result.is_empty(),
            request_id=request_id() or None,
        )

    return app


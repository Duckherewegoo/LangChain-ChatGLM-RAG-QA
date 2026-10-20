"""FastAPI 应用与 HTTP 路由。

这一层刻意保持极薄：校验传输契约、附加可观测性上下文、委托业务逻辑给容器。
它不导入提供者 SDK，也不实现检索算法。
"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from api.handlers import (
    _build_health,
    _build_model_health,
    _ensure_vector_store,
    _run_agent,
    _sse,
    _to_answer_request,
)
from api.middleware import map_exception
from api.schemas import (
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
from exceptions import RAGError, ResourceNotFoundError, ValidationError
from observability import (
    collect_metrics,
    configure_observability,
    finish_request,
    get_logger,
    get_metric,
    request_id,
    start_request,
)
from rag.agent import AgentRequest
from rag.agent_events import AgentEventType
from rag.container import Container, get_container
from rag.service import AnswerRequest
from settings import load_settings

logger = get_logger(__name__)


def create_app(container: Container | None = None) -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    settings = load_settings()
    configure_observability(settings.log_level, settings.observability.metrics_enabled)
    dependencies = container or get_container()

    app = FastAPI(
        title="Enterprise RAG Assistant",
        version="0.1.0",
        description="基于 LangChain/LangGraph 的 RAG 服务，支持 ChatGLM/Qwen/DeepSeek 多模型。",
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
        request_latency = get_metric("request_latency")
        requests_total = get_metric("requests_total")

        try:
            response = await call_next(request)
        except Exception as exc:
            status_code, envelope = map_exception(exc, request_id=correlation_id)
            response = JSONResponse(status_code=status_code, content=envelope.model_dump())
        finally:
            elapsed = finish_request()

        response.headers["X-Request-Id"] = correlation_id
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        if request_latency is not None:
            request_latency.labels(method=request.method, endpoint=endpoint).observe(elapsed)
        if requests_total is not None:
            requests_total.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
            ).inc()
        return response

    @app.exception_handler(Exception)
    async def _fallback_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        status_code, envelope = map_exception(exc, request_id=request_id() or None)
        return JSONResponse(status_code=status_code, content=envelope.model_dump())

    @app.exception_handler(RAGError)
    async def _domain_exception_handler(request: Request, exc: RAGError) -> JSONResponse:
        status_code, envelope = map_exception(exc, request_id=request_id() or None)
        return JSONResponse(status_code=status_code, content=envelope.model_dump())

    # ---- 健康检查 ----------------------------------------------------------

    @app.get("/healthz", response_model=HealthResponseBody)
    async def healthz() -> HealthResponseBody:
        """存活探针：只检查进程存活，不触碰外部依赖。

        liveness 失败会被 K8s 重启，所以这里绝不能阻塞在向量库或模型连接上。
        """
        return _build_health(dependencies, liveness=True)

    @app.get("/readyz", response_model=HealthResponseBody)
    async def readyz() -> HealthResponseBody:
        """就绪探针：真正触发向量库与嵌入模型的初始化，检查一致性。"""
        _ensure_vector_store(dependencies)
        return _build_health(dependencies, liveness=False)

    @app.get("/healthz/models", response_model=HealthResponseBody)
    async def models_health() -> HealthResponseBody:
        return await _build_model_health(dependencies)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint() -> Response:
        payload = collect_metrics()
        if payload is None:
            return PlainTextResponse("# Prometheus client 未安装\n", media_type="text/plain")
        return PlainTextResponse(
            payload.decode("utf-8"), media_type="text/plain; version=0.0.4"
        )

    # ---- 回答端点 ----------------------------------------------------------

    @app.post("/v1/answer", response_model=AnswerResponseBody)
    async def create_answer(body: AnswerRequestBody) -> AnswerResponseBody:
        dependencies.guardrails().check_question(body.question)
        service = dependencies.rag_service()
        request = _to_answer_request(body)

        result = await service.aanswer(request)
        return AnswerResponseBody(
            answer=result.answer,
            query=result.query,
            citations=[CitationView(**c.model_dump()) for c in result.citations],
            documents=[
                RetrievedDocumentView(**d.model_dump())
                for d in result.context.documents
            ],
            context_used=result.context_used,
            model=result.model,
            tokens=result.tokens,
            attempted_models=result.attempted_models,
            fallback_used=result.fallback_used,
            request_id=request_id() or None,
        )

    @app.get("/v1/retrieve", response_model=AnswerResponseBody)
    async def retrieve(
        question: str,
        tenant_id: str = "default",
        top_k: int = 8,
        owner_id: str | None = None,
    ) -> AnswerResponseBody:
        dependencies.guardrails().check_question(question)
        service = dependencies.rag_service()
        result = service.retrieve_only(
            AnswerRequest(
                question=question,
                tenant_id=tenant_id,
                owner_id=owner_id,
                top_k=top_k,
            )
        )
        return AnswerResponseBody(
            answer="",
            query=question,
            documents=[
                RetrievedDocumentView(**d.model_dump()) for d in result.documents
            ],
            context_used=not result.is_empty(),
            request_id=request_id() or None,
        )

    # ---- Agent 端点 --------------------------------------------------------

    @app.post("/v1/agent", response_model=AgentResponseBody)
    async def run_agent(body: AgentRequestBody) -> AgentResponseBody:
        dependencies.guardrails().check_question(body.question)
        runtime_request = AgentRequest(
            question=body.question.strip(),
            tenant_id=body.tenant_id,
            owner_id=body.owner_id,
            allowed_sources=body.allowed_sources,
            history=body.history,
            additional_context=body.additional_context,
            use_case=body.use_case,
        )
        response = await _run_agent(
            dependencies, runtime_request, body.additional_context
        )

        # trail_store 是 Container 的属性，不是 AgentRunner 的
        dependencies.trail_store.put(response.request_id, response.trail)

        return AgentResponseBody(
            request_id=response.request_id,
            answer=response.answer,
            citations=response.citations,
            tool_calls=[
                AgentToolCallView(**call.model_dump()) for call in response.tool_calls
            ],
            iterations=response.iterations,
            stopped_reason=response.stopped_reason,
            model=response.model,
            documents=response.documents,
            attempted_models=response.attempted_models,
            fallback_used=response.fallback_used,
        )

    @app.post("/v1/agent/stream")
    async def stream_agent(body: AgentRequestBody):
        dependencies.guardrails().check_question(body.question)
        runtime_request = AgentRequest(
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
            yield _sse(
                {
                    "event": AgentEventType.STARTED.value,
                    "request_id": runtime_request.request_id,
                }
            )
            try:
                # runner.stream 是同步生成器
                for event in runner.stream(runtime_request):
                    yield _sse(event)
            except ValidationError as exc:
                yield _sse(
                    {
                        "event": "error",
                        "error_code": exc.error_code,
                        "message": exc.message,
                    }
                )
            except Exception as exc:
                _code, envelope = map_exception(
                    exc, request_id=runtime_request.request_id
                )
                yield _sse({"event": "error", **envelope.model_dump()})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/v1/agent/{agent_request_id}/trail", response_class=PlainTextResponse)
    async def get_agent_trail(agent_request_id: str) -> Response:
        """返回请求的完整 JSONL 审计轨迹。"""
        store = dependencies.trail_store
        trail = store.get(agent_request_id)
        if trail is None:
            raise ResourceNotFoundError(
                f"未找到请求 {agent_request_id!r} 的审计轨迹。"
            )
        return PlainTextResponse(trail.to_jsonl(), media_type="application/x-ndjson")

    # ---- 摄入与文档管理 ---------------------------------------------------

    @app.post("/v1/ingest", response_model=IngestResponseBody)
    async def ingest(body: IngestRequestBody) -> IngestResponseBody:
        pipeline = dependencies.ingest_pipeline()
        stats = pipeline.ingest_path(
            body.path,
            tenant_id=body.tenant_id,
            owner_id=body.owner_id,
            embedder=dependencies.embedder(),
            title=body.title,
            tags=body.tags,
        )
        return IngestResponseBody(
            status="accepted",
            message=f"已从 {stats.scanned} 个文件摄入 {stats.chunks_stored} 个文本块。",
            document_id=stats.document_id,
            scanned=stats.scanned,
            chunks_stored=stats.chunks_stored,
            bytes_size=stats.bytes_size,
            request_id=request_id() or None,
        )

    @app.delete("/v1/documents/{document_id}", response_model=MessageResponseBody)
    async def delete_document(
        document_id: str, tenant_id: str = "default"
    ) -> MessageResponseBody:
        pipeline = dependencies.ingest_pipeline()
        removed = pipeline.remove_document(document_id, tenant_id)
        return MessageResponseBody(
            status="deleted",
            message=f"已删除文档 {document_id} 的 {removed} 个文本块。",
            request_id=request_id() or None,
        )

    return app

"""FastAPI 应用与 HTTP 路由。

这一层刻意保持极薄：校验传输契约、附加可观测性上下文、委托业务逻辑给容器。
它不导入提供者 SDK，也不实现检索算法。

两条问答路径并存：
    - /v1/answer、/v1/agent → 手写多角色编排器（含审核，无 LangGraph）；
    - /v1/agent/stream → LangGraph 智能体（无审核，流式输出）。
二者互不调用。需要审核走 /v1/agent；需要流式走 /v1/agent/stream。

安全说明（待补齐）：
    /v1/ingest 接收客户端提供的文件路径，未做路径白名单校验；
    /v1/documents DELETE 未做鉴权，tenant_id 由客户端填写。
    暴露到公网前必须补齐——本文件只标注，不实现。
"""

from __future__ import annotations
import asyncio
import typing
import uuid
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from api.handlers import (
    _build_health,
    _build_model_health,
    _ensure_vector_store,
    _run_via_orchestrator,
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
from base.exceptions import RAGError, ResourceNotFoundError, ValidationError
from base.observability import (
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


def _package_version() -> str:
    """读取包版本；包未安装时回退到开发版本号。

    模块顶层直接 version() 在未 pip install -e 的开发环境会抛
    PackageNotFoundError，导致 import api.server 即崩。
    """
    try:
        return version("langchain-chatglm-rag-qa")
    except PackageNotFoundError:
        return "0.0.0-dev"


value = _package_version()


_CITATION_ALLOWED_FIELDS = {"index", "source", "title", "page", "chunk_id", "score"}


def _citation_view(item: typing.Any) -> CitationView:
    """把 pydantic 对象或 dict 统一映射为 CitationView。

    /v1/answer 的 service 返回 pydantic 对象，/v1/agent 的编排器返回 dict，
    两条路径此前各自处理，容易漂移。统一在这里做白名单过滤，避免字段
    不匹配导致的 500。
    """
    raw = item.model_dump() if hasattr(item, "model_dump") else dict(item)
    return CitationView(**{k: v for k, v in raw.items() if k in _CITATION_ALLOWED_FIELDS})


def _to_agent_request(body: AgentRequestBody) -> AgentRequest:
    """把 API 请求映射为 AgentRequest（/v1/agent 与 /v1/agent/stream 共用）。

    request_id 透传中间件捕获的 X-Request-Id；缺失时生成 uuid4，保证
    /v1/agent 的 trail 与客户端日志使用同一个 ID。
    """
    return AgentRequest(
        question=body.question.strip(),
        tenant_id=body.tenant_id,
        owner_id=body.owner_id,
        allowed_sources=body.allowed_sources,
        history=body.history,
        additional_context=body.additional_context,
        use_case=body.use_case,
        request_id=request_id() or uuid.uuid4().hex,
    )


def _next_or_none(iterator: typing.Iterator[typing.Any]) -> tuple[bool, typing.Any]:
    """从同步迭代器拉一个元素；结束返回 (False, None)。

    用于把同步生成器包装成异步流；asyncio.to_thread 不接受 StopIteration，
    所以用哨兵把「有值/结束」显式区分。
    """
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def create_app(container: Container | None = None) -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    # 注（M3）：get_container 内部也会 load_settings()；此处先 load 是为了
    # 在容器初始化前配置 observability。若 container 已传入则复用其 settings。
    settings = container.settings if container is not None else load_settings()
    configure_observability(settings.log_level, settings.observability.metrics_enabled)
    dependencies = container or get_container()

    app = FastAPI(
        title="Enterprise RAG Assistant",
        version=value,
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
    async def request_context_middleware(
        request: Request,
        call_next: typing.Callable[[Request], typing.Awaitable[Response]],
    ) -> Response:
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
            # 503 让 K8s / Prometheus 认为端点不健康，避免监控盲区
            return PlainTextResponse(
                "# Prometheus client 未安装\n",
                status_code=503,
                media_type="text/plain",
            )
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
            citations=[_citation_view(c) for c in result.citations],
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
        top_k: int | None = None,
        owner_id: str | None = None,
    ) -> AnswerResponseBody:
        dependencies.guardrails().check_question(question)
        service = dependencies.rag_service()
        # top_k 未指定时用 settings 默认，避免与 RetrievalSettings.top_k 漂移
        effective_top_k = (
            top_k if top_k is not None else dependencies.settings.retrieval.top_k
        )
        result = service.retrieve_only(
            AnswerRequest(
                question=question,
                tenant_id=tenant_id,
                owner_id=owner_id,
                top_k=effective_top_k,
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
        runtime_request = _to_agent_request(body)
        response = await _run_via_orchestrator(
            dependencies, runtime_request, body.additional_context
        )

        # trail_store 是 Container 的属性，不是 AgentRunner 的
        # 注（P1-4）：只有本端点写 trail_store；/v1/answer 与 /v1/agent/stream
        # 不写，因此 get_agent_trail 对后两者返回 404。
        dependencies.trail_store.put(response.request_id, response.trail)

        return AgentResponseBody(
            request_id=response.request_id,
            answer=response.answer,
            citations=[_citation_view(item) for item in response.citations],
            tool_calls=[
                AgentToolCallView(**call.model_dump()) for call in response.tool_calls
            ],
            # 注（M6）：orchestrator 不调工具，tool_calls 恒为空；
            # attempted_models / fallback_used 也恒为默认值（见 handlers 的 TODO）。
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
        runtime_request = _to_agent_request(body)
        runner = dependencies.agent_runner()

        async def event_stream():
            yield _sse(
                {
                    "event": AgentEventType.STARTED.value,
                    "request_id": runtime_request.request_id,
                }
            )
            # P0-4：runner.stream 是同步生成器，直接在 async 里 for 会阻塞事件循环。
            # 用 asyncio.to_thread 逐条拉取，把「计算下一条」丢到线程池。
            iterator = runner.stream(runtime_request)
            try:
                while True:
                    has_next, event = await asyncio.to_thread(_next_or_none, iterator)
                    if not has_next:
                        break
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
                # P1-5：SSE 无 HTTP status，_code 被丢弃，用 error_code 替代
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
        # P0-2 待补：body.path 未做路径白名单校验，暴露公网前必须修。
        # M4：ingest_path 是同步大文件操作，提交到线程池，避免阻塞事件循环。
        stats = await asyncio.to_thread(
            pipeline.ingest_path,
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
        # P0-3 待补：无鉴权，tenant_id 由客户端填写，暴露公网前必须修。
        removed = await asyncio.to_thread(
            pipeline.remove_document, document_id, tenant_id
        )
        return MessageResponseBody(
            status="deleted",
            message=f"已删除文档 {document_id} 的 {removed} 个文本块。",
            request_id=request_id() or None,
        )

    return app

"""可观测性基础设施：基于 Loguru 的统一日志与指标。"""

from __future__ import annotations

import logging
import sys
import time
import typing
import uuid
from contextvars import ContextVar

from loguru import logger as loguru_logger

# Prometheus 指标（优雅降级）
try:
    from prometheus_client import Counter, Histogram, REGISTRY, generate_latest
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    REGISTRY = None  # type: ignore
    generate_latest = None  # type: ignore
    _PROMETHEUS_AVAILABLE = False

# 请求上下文
_REQUEST_CONTEXT: ContextVar[dict[str, typing.Any]] = ContextVar(
    "request_context", default={}
)

# 指标注册表
_METRIC_REGISTRY: dict[str, typing.Any] = {}


# ============================================================
# 日志配置（Loguru 接管一切）
# ============================================================

class InterceptHandler(logging.Handler):
    """将标准 logging 日志桥接到 Loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_observability(log_level: str = "INFO", metrics_enabled: bool = True) -> None:
    """配置 Loguru 与 Prometheus 指标。"""
    loguru_logger.remove()
    loguru_logger.add(
        sys.stdout,
        level=log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[request_id]}</cyan> | "
            "<cyan>{extra[module]}</cyan> | "
            "<level>{message}</level>"
        ),
        enqueue=True,
        serialize=False,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    if metrics_enabled:
        _initialize_metrics()


def get_logger(name: str | None = None) -> typing.Any:
    """返回一个绑定了模块名的 Loguru logger。"""
    return loguru_logger.bind(
        module=name or "root",
        request_id=_REQUEST_CONTEXT.get().get("request_id", "-"),
    )


# ============================================================
# 请求上下文
# ============================================================

def start_request(request_id: str | None = None) -> str:
    """为当前请求绑定关联标识符。"""
    value = request_id or uuid.uuid4().hex
    _REQUEST_CONTEXT.set({
        "request_id": value,
        "start_time": time.perf_counter(),
    })
    return value


def request_id() -> str:
    """返回当前请求的关联标识符。"""
    return _REQUEST_CONTEXT.get().get("request_id", "")


def finish_request() -> float:
    """清理请求上下文并返回耗时（秒）。"""
    ctx = _REQUEST_CONTEXT.get()
    start = ctx.get("start_time", 0.0)
    elapsed = time.perf_counter() - start
    _REQUEST_CONTEXT.set({})
    return elapsed


def request_scope(request_id_value: str | None = None) -> typing.Iterator[str]:
    """上下文管理器形式管理请求生命周期。"""
    rid = start_request(request_id_value)
    try:
        yield rid
    finally:
        finish_request()


# ============================================================
# Prometheus 指标
# ============================================================

def _create_metric(name: str, factory: typing.Callable[[], typing.Any]) -> typing.Any:
    if name in _METRIC_REGISTRY:
        return _METRIC_REGISTRY[name]
    metric = factory()
    _METRIC_REGISTRY[name] = metric
    return metric


def _initialize_metrics() -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    _create_metric("requests_total", lambda: Counter(
        "rag_http_requests_total", "Total HTTP requests.", ["method", "endpoint", "status_code"]
    ))
    _create_metric("request_latency", lambda: Histogram(
        "rag_http_request_duration_seconds", "HTTP request latency.", ["method", "endpoint"]
    ))
    _create_metric("retrieval_documents", lambda: Histogram(
        "rag_retrieved_documents_total", "Documents retrieved per query.", ["tenant_id"]
    ))
    _create_metric("tool_calls_total", lambda: Counter(
        "rag_agent_tool_calls_total", "Tool invocations.", ["tool", "status"]
    ))
    _create_metric("ingested_chunks", lambda: Counter(
        "rag_ingested_chunks_total", "Chunks ingested.", ["source"]
    ))
    _create_metric("rerank_latency", lambda: Histogram(
        "rag_rerank_duration_seconds", "Reranking latency."
    ))


def get_metric(name: str) -> typing.Any:
    """获取已初始化的 Prometheus 指标，未初始化或不存在时返回 None。"""
    return _METRIC_REGISTRY.get(name)


def collect_metrics() -> bytes | None:
    if generate_latest is None:
        return None
    return generate_latest()


def metrics_available() -> bool:
    return _PROMETHEUS_AVAILABLE


def shutdown_observability() -> None:
    if _PROMETHEUS_AVAILABLE and REGISTRY is not None:
        for metric in list(_METRIC_REGISTRY.values()):
            try:
                REGISTRY.unregister(metric)
            except (KeyError, ValueError):
                pass
    _METRIC_REGISTRY.clear()
    _REQUEST_CONTEXT.set({})


# ============================================================
# 计时器
# ============================================================

class timed:
    """同步/异步计时器，上报 Prometheus 指标。"""

    def __init__(self, metric: typing.Any | None, **labels: str) -> None:
        self._metric = metric
        self._labels = labels
        self._started = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "timed":
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._started
        if self._metric is not None:
            try:
                self._metric.labels(**self._labels).observe(self.elapsed)
            except Exception:
                pass

    async def __aenter__(self) -> "timed":
        self._started = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._started
        if self._metric is not None:
            try:
                self._metric.labels(**self._labels).observe(self.elapsed)
            except Exception:
                pass


__all__ = [
    "configure_observability",
    "get_logger",
    "get_metric",
    "start_request",
    "request_id",
    "finish_request",
    "request_scope",
    "collect_metrics",
    "metrics_available",
    "shutdown_observability",
    "timed",
]

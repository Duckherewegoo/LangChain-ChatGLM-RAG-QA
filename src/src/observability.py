"""Observability primitives.

This module centralizes logging, request correlation, and Prometheus metrics.
It intentionally avoids making the core pipeline dependent on a particular
tracing vendor; integrations can be enabled through ``configure_observability``.
"""

from __future__ import annotations

import logging
import time
import typing
import uuid
from contextvars import ContextVar

import structlog

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Iterator

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter, Histogram, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - library absent during some tests
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    _PROMETHEUS_AVAILABLE = False

_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
_START_TIME: ContextVar[float] = ContextVar("start_time", default=0.0)

REQUESTS_TOTAL: Counter | None = None
REQUEST_LATENCY: Histogram | None = None
RETRIEVAL_DOCUMENTS: Histogram | None = None
TOOL_CALLS_TOTAL: Counter | None = None
INGESTED_CHUNKS: Counter | None = None
RERANK_LATENCY: Histogram | None = None

_REQUESTS_TOTAL_DESC = "Total HTTP requests processed by the RAG service."
_REQUEST_LATENCY_DESC = "HTTP request latency in seconds."
_RETRIEVAL_DOCUMENTS_DESC = "Number of documents retrieved per query."
_TOOL_CALLS_TOTAL_DESC = "Total agent tool invocations."
_INGESTED_CHUNKS_DESC = "Total document chunks ingested."
_RERANK_LATENCY_DESC = "Reranking latency in seconds."


def configure_observability(log_level: str = "INFO", metrics_enabled: bool = True) -> None:
    """Configure logging and, when available, Prometheus metrics."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[logging.StreamHandler()],
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if metrics_enabled and _PROMETHEUS_AVAILABLE:
        _initialize_metrics()


def _initialize_metrics() -> None:
    global REQUESTS_TOTAL, REQUEST_LATENCY, RETRIEVAL_DOCUMENTS, TOOL_CALLS_TOTAL
    global INGESTED_CHUNKS, RERANK_LATENCY

    if REQUESTS_TOTAL is not None:
        return

    REQUESTS_TOTAL = Counter(
        "rag_http_requests_total", _REQUESTS_TOTAL_DESC, ["method", "endpoint", "status_code"]
    )
    REQUEST_LATENCY = Histogram(
        "rag_http_request_duration_seconds", _REQUEST_LATENCY_DESC, ["method", "endpoint"]
    )
    RETRIEVAL_DOCUMENTS = Histogram(
        "rag_retrieved_documents_total", _RETRIEVAL_DOCUMENTS_DESC, ["tenant_id"]
    )
    TOOL_CALLS_TOTAL = Counter(
        "rag_agent_tool_calls_total", _TOOL_CALLS_TOTAL_DESC, ["tool", "status"]
    )
    INGESTED_CHUNKS = Counter("rag_ingested_chunks_total", _INGESTED_CHUNKS_DESC, ["source"])
    RERANK_LATENCY = Histogram("rag_rerank_duration_seconds", _RERANK_LATENCY_DESC)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a context-aware structured logger."""

    return structlog.get_logger(name)


def start_request(request_id: str | None = None) -> str:
    """Bind a correlation identifier for the active request."""

    value = request_id or uuid.uuid4().hex
    _REQUEST_ID.set(value)
    _START_TIME.set(time.perf_counter())
    structlog.contextvars.bind_contextvars(request_id=value)
    return value


def request_id() -> str:
    """Return the current correlation identifier."""

    return _REQUEST_ID.get()


def finish_request() -> float:
    """Clear the request context and return elapsed seconds."""

    elapsed = time.perf_counter() - _START_TIME.get()
    structlog.contextvars.unbind_contextvars("request_id")
    _REQUEST_ID.set("")
    return elapsed


def collect_metrics() -> bytes | None:
    """Render Prometheus metrics when the client is installed."""

    if generate_latest is None:
        return None
    return generate_latest()


class timed:
    """Context manager that records elapsed seconds without swallowing errors."""

    def __init__(self, metric: Histogram | None, **labels: str) -> None:
        self._metric = metric
        self._labels = labels
        self._started = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> timed:
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._started
        if self._metric is not None:
            self._metric.labels(**self._labels).observe(self.elapsed)


def metrics_available() -> bool:
    """Indicate whether Prometheus metrics support is installed."""

    return _PROMETHEUS_AVAILABLE


def metrics_context() -> Iterator[None]:
    """Yield once while exporting metrics availability for testing."""

    yield None

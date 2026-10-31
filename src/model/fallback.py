"""模型回退执行。

本模块实现顺序回退链。对于首选模型的瞬时故障进行重试，
随后升级至备用提供者。永久性故障（如认证、内容策略、配置错误）
不会被掩盖，因为它们在所有提供者上都会重复发生。
"""

from __future__ import annotations

import asyncio
import time
import typing

from pydantic import BaseModel, Field

from model.protocol import ChatModel, ChatRequest, ChatResponse
from base.observability import get_logger

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from collections.abc import Iterable

logger = get_logger(__name__)


class FallbackAttempt(BaseModel):
    """回退链中单次尝试的记录。"""

    provider: str
    model_name: str
    status: str
    error_type: str | None = None
    error_message: str | None = None
    retryable: bool = False
    elapsed_seconds: float = 0.0
    response: ChatResponse | None = None


class FallbackResult(BaseModel):
    """执行回退链的结果。"""

    response: ChatResponse | None = None
    succeeded: bool = False
    attempts: list[FallbackAttempt] = Field(default_factory=list)
    final_error: str | None = None
    error_types: list[str] = Field(
        default_factory=list, description="所有失败尝试的错误类型，去重保序"
    )

    def summary(self) -> dict[str, typing.Any]:
        """返回用于日志和诊断的可序列化摘要。"""
        return {
            "succeeded": self.succeeded,
            "attempts": [attempt.model_dump(exclude={"response"}) for attempt in self.attempts],
            "final_error": self.final_error,
            "error_types": self.error_types,
        }

    @property
    def fallback_used(self) -> bool:
        """成功的提供者是否不同于首次尝试的提供者（即发生跨提供者回退）。"""
        successful = [attempt for attempt in self.attempts if attempt.status == "success"]
        return bool(successful) and successful[0].provider != self.attempts[0].provider


class FallbackExecutor:
    """在优先级链上执行模型请求。

    max_attempts_per_provider 是每个提供者的尝试次数，不是整条链的总次数；
    需要总预算时传入 max_total_attempts。
    """

    def __init__(
        self,
        primary: ChatModel,
        secondaries: Iterable[ChatModel] = (),
        *,
        max_attempts_per_provider: int = 1,
        max_total_attempts: int | None = None,
    ) -> None:
        if max_attempts_per_provider < 1:
            raise ValueError("每个提供者的最大尝试次数至少为 1。")
        if max_total_attempts is not None and max_total_attempts < 1:
            raise ValueError("总尝试次数上限至少为 1。")

        self._chain: list[ChatModel] = [primary, *secondaries]
        self._max_attempts = max_attempts_per_provider
        self._max_total = max_total_attempts

    @property
    def providers(self) -> list[str]:
        """按执行顺序返回提供者标识符。"""
        return [model.name for model in self._chain]

    def invoke(self, request: ChatRequest) -> FallbackResult:
        """在回退链上同步执行请求。"""
        result = FallbackResult()
        total = 0
        for model in self._chain:
            for attempt_index in range(self._max_attempts):
                if self._max_total is not None and total >= self._max_total:
                    return _finalize(result)
                total += 1
                attempt = self._try_once(model, request, attempt_index)
                result.attempts.append(attempt)

                if attempt.status == "success":
                    result.succeeded = True
                    result.response = attempt.response
                    return result

                if not attempt.retryable:
                    break
                if attempt_index < self._max_attempts - 1:
                    time.sleep(_backoff_seconds(attempt_index))

        return _finalize(result)

    async def ainvoke(self, request: ChatRequest) -> FallbackResult:
        """在回退链上异步执行请求。"""
        result = FallbackResult()
        total = 0
        for model in self._chain:
            for attempt_index in range(self._max_attempts):
                if self._max_total is not None and total >= self._max_total:
                    return _finalize(result)
                total += 1
                attempt = await self._try_once_async(model, request, attempt_index)
                result.attempts.append(attempt)

                if attempt.status == "success":
                    result.succeeded = True
                    result.response = attempt.response
                    return result

                if not attempt.retryable:
                    break
                if attempt_index < self._max_attempts - 1:
                    await asyncio.sleep(_backoff_seconds(attempt_index))

        return _finalize(result)

    def _try_once(self, model: ChatModel, request: ChatRequest, attempt_index: int) -> FallbackAttempt:
        started = time.perf_counter()
        try:
            response = model.invoke(request)
            return FallbackAttempt(
                provider=model.name,
                model_name=model.model_name,
                status="success",
                elapsed_seconds=time.perf_counter() - started,
                response=response,
            )
        except Exception as exc:
            return self._record_failure(model, exc, started, attempt_index)

    async def _try_once_async(
        self, model: ChatModel, request: ChatRequest, attempt_index: int
    ) -> FallbackAttempt:
        started = time.perf_counter()
        try:
            response = await model.ainvoke(request)
            return FallbackAttempt(
                provider=model.name,
                model_name=model.model_name,
                status="success",
                elapsed_seconds=time.perf_counter() - started,
                response=response,
            )
        except Exception as exc:
            return self._record_failure(model, exc, started, attempt_index)

    @staticmethod
    def _record_failure(
        model: ChatModel, exc: BaseException, started: float, attempt_index: int
    ) -> FallbackAttempt:
        # 局部导入防止顶层的循环依赖
        from model.errors import classify_model_error

        classified = classify_model_error(exc, provider=model.name)
        model_name = model.model_name

        logger.warning(
            "模型回退尝试失败 | provider=%s | model=%s | attempt=%s | type=%s | retryable=%s | error=%s",
            model.name,
            model_name,
            attempt_index + 1,
            classified.error_type,
            classified.retryable,
            exc,
        )

        return FallbackAttempt(
            provider=model.name,
            model_name=model_name,
            status="failed",
            error_type=classified.error_type,
            error_message=str(exc),
            retryable=classified.retryable,
            elapsed_seconds=time.perf_counter() - started,
        )


def _backoff_seconds(attempt_index: int) -> float:
    """指数退避：1s、2s、4s，封顶 4s。"""
    return min(2.0 ** attempt_index, 4.0)


def _finalize(result: FallbackResult) -> FallbackResult:
    """收尾：计算 final_error 与 error_types。"""
    result.final_error = (
        result.attempts[-1].error_message if result.attempts else "无可用模型"
    )
    seen: list[str] = []
    for attempt in result.attempts:
        if attempt.status != "success" and attempt.error_type and attempt.error_type not in seen:
            seen.append(attempt.error_type)
    result.error_types = seen
    return result


__all__ = ["FallbackAttempt", "FallbackExecutor", "FallbackResult"]

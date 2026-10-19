"""模型回退执行。

本模块实现顺序回退链。对于首选模型的瞬时故障进行重试，
随后升级至备用提供者。永久性故障（如认证、内容策略、配置错误）
不会被掩盖，因为它们在所有提供者上都会重复发生。
"""

from __future__ import annotations

import time
import typing

from pydantic import BaseModel, Field

from src.model.protocol import ChatModel, ChatRequest, ChatResponse
from src.observability import get_logger

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

    def summary(self) -> dict[str, typing.Any]:
        """返回用于日志和诊断的可序列化摘要。"""
        return {
            "succeeded": self.succeeded,
            "attempts": [attempt.model_dump(exclude={"response"}) for attempt in self.attempts],
            "final_error": self.final_error,
        }


class FallbackExecutor:
    """在优先级链上执行模型请求。"""

    def __init__(
        self,
        primary: ChatModel,
        secondaries: Iterable[ChatModel] = (),
        *,
        max_attempts_per_provider: int = 1,
    ) -> None:
        if max_attempts_per_provider < 1:
            raise ValueError("每个提供者的最大尝试次数至少为 1。")

        self._chain: list[ChatModel] = [primary, *secondaries]
        if not self._chain:
            raise ValueError("回退链至少需要包含一个模型。")
        self._max_attempts = max_attempts_per_provider

    @property
    def providers(self) -> list[str]:
        """按执行顺序返回提供者标识符。"""
        return [model.name for model in self._chain]

    def invoke(self, request: ChatRequest) -> FallbackResult:
        """在回退链上同步执行请求。"""
        result = FallbackResult()
        for model in self._chain:
            for attempt_index in range(self._max_attempts):
                attempt = self._try_once(model, request, attempt_index)
                result.attempts.append(attempt)

                if attempt.status == "success":
                    result.succeeded = True
                    result.response = attempt.response
                    return result

                if not attempt.retryable:
                    break

        result.final_error = result.attempts[-1].error_message if result.attempts else "无可用模型"
        return result

    async def ainvoke(self, request: ChatRequest) -> FallbackResult:
        """在回退链上异步执行请求。"""
        result = FallbackResult()
        for model in self._chain:
            for attempt_index in range(self._max_attempts):
                attempt = await self._try_once_async(model, request, attempt_index)
                result.attempts.append(attempt)

                if attempt.status == "success":
                    result.succeeded = True
                    result.response = attempt.response
                    return result

                if not attempt.retryable:
                    break

        result.final_error = result.attempts[-1].error_message if result.attempts else "无可用模型"
        return result

    def _try_once(self, model: ChatModel, request: ChatRequest, attempt_index: int) -> FallbackAttempt:
        started = time.perf_counter()
        try:
            response = model.invoke(request)
            return FallbackAttempt(
                provider=model.name,
                model_name=getattr(model, "model_name", model.name),
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
                model_name=getattr(model, "model_name", model.name),
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
        from src.model.errors import classify_model_error

        classified = classify_model_error(exc, provider=model.name)
        model_name = getattr(model, "model_name", model.name)

        # 将 kwargs 风格改为标准的 logging 参数插值，避免底层 logger 不兼容报错
        logger.warning(
            f"模型回退尝试失败 | provider={model.name} | model={model_name} | "
            f"attempt={attempt_index + 1} | type={classified.error_type} | "
            f"retryable={classified.retryable} | error={exc!s}"
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


__all__ = ["FallbackAttempt", "FallbackExecutor", "FallbackResult"]

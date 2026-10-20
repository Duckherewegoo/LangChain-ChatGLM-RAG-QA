"""兼容 OpenAI 协议的对话模型适配器。

本适配器针对实现了 OpenAI Chat Completions 协议的端点，
包括 DeepSeek、Qwen/DashScope 兼容模式、智谱 AI 兼容端点以及任意提供相同契约的网关。
特定厂商的模块可以通过继承它来调整默认值或归一化特有响应。

实现上刻意直接使用 httpx，而不引入更重的链式包装器，
从而保持依赖权重和故障模式的显式可控。
"""

from __future__ import annotations

import json
import time
import typing

import httpx

from model.errors import (
    ModelAPIConnectionError,
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelContentPolicyError,
    ModelPayloadError,
    ModelRateLimitError,
    ModelTimeoutError,
    is_retryable,
)
from model.protocol import ChatModel, ChatRequest, ChatResponse
from observability import get_logger

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from settings import ModelChatSettings

logger = get_logger(__name__)


class _OpenAICompatibleChatModel:
    """兼容 OpenAI 协议的对话端点基础适配器。"""

    name: str = "openai_compatible"
    model_name: str = ""
    DEFAULT_BASE_URL = ""
    DEFAULT_MAX_RETRIES = 1

    def __init__(self, settings: ModelChatSettings) -> None:
        self._settings = settings
        self.model_name = settings.model_name
        self._assert_configured()

        # 同步客户端实例化
        self._client = httpx.Client(timeout=settings.timeout_seconds)
        # 异步客户端也实例化在这里，避免每次请求重新建连
        self._async_client = httpx.AsyncClient(timeout=settings.timeout_seconds)
        self._bound_tools: list[dict[str, typing.Any]] | None = None

    def _assert_configured(self) -> None:
        if not self._settings.is_configured:
            raise ModelConfigurationError(
                "对话模型 API Key 为空；请配置 MODEL_CHAT_API_KEY 或 settings.yaml。",
                details={
                    "provider": self._settings.provider,
                    "model_name": self._settings.model_name,
                },
            )

    @property
    def _endpoint(self) -> str:
        base_url = (self._settings.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        return f"{base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"enterprise-rag-chatglm/{self.name}",
        }

    def _payload(
        self,
        request: ChatRequest,
        *,
        tools: typing.Sequence[dict[str, typing.Any]] | None = None,
        tool_choice: str | dict[str, typing.Any] | None = None,
    ) -> dict[str, typing.Any]:

        # 修复：手动转换消息格式，不再依赖未实现的 ChatRequest 方法
        openai_messages = []
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            item = {"role": role, "content": content}
            if "tool_calls" in msg:
                item["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                item["tool_call_id"] = msg["tool_call_id"]
            openai_messages.append(item)

        payload: dict[str, typing.Any] = {
            "model": self._settings.model_name,
            "messages": openai_messages,
            "temperature": self._settings.temperature,
        }
        if self._settings.max_tokens:
            payload["max_tokens"] = self._settings.max_tokens
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        return payload

    def bind_tools(self, tools: typing.Sequence[dict[str, typing.Any]]) -> ChatModel:
        self._bound_tools = [dict(tool) for tool in tools]
        return self

    def invoke(self, request: ChatRequest) -> ChatResponse:
        return self._request_with_retry(request)

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return await self._arequest_with_retry(request)

    def _request_with_retry(self, request: ChatRequest) -> ChatResponse:
        attempts = 0
        maximum = max(1, self._settings.max_retries + 1)
        while True:
            attempts += 1
            try:
                return self._post(request)
            except Exception as exc:
                if attempts >= maximum or not is_retryable(exc, provider=self._settings.provider):
                    raise
                delay = min(2 ** (attempts - 1), 8)
                # 修复：改为标准的 logging 格式化
                logger.warning(
                    f"模型重试已调度 | provider={self._settings.provider} | "
                    f"attempt={attempts} | delay={delay}s | error={exc!s}"
                )
                time.sleep(delay)

    async def _arequest_with_retry(self, request: ChatRequest) -> ChatResponse:
        attempts = 0
        maximum = max(1, self._settings.max_retries + 1)
        while True:
            attempts += 1
            try:
                return await self._apost(request)
            except Exception as exc:
                if attempts >= maximum or not is_retryable(exc, provider=self._settings.provider):
                    raise
                delay = min(2 ** (attempts - 1), 8)
                logger.warning(
                    f"模型重试已调度(异步) | provider={self._settings.provider} | "
                    f"attempt={attempts} | delay={delay}s | error={exc!s}"
                )
                await _async_sleep(delay)

    def _post(self, request: ChatRequest) -> ChatResponse:
        payload = self._payload(
            request,
            tools=self._bound_tools,
            tool_choice="auto" if self._bound_tools else None,
        )
        started = time.perf_counter()
        try:
            response = self._client.post(
                self._endpoint,
                json=payload,
                headers=self._headers(),
            )
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(
                f"模型请求超时 ({time.perf_counter() - started:.2f}s): {exc}",
                details={"provider": self._settings.provider},
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelAPIConnectionError(
                f"模型连接失败: {exc}",
                details={"provider": self._settings.provider},
            ) from exc

        return self._parse_response(response, latency=time.perf_counter() - started)

    async def _apost(self, request: ChatRequest) -> ChatResponse:
        payload = self._payload(
            request,
            tools=self._bound_tools,
            tool_choice="auto" if self._bound_tools else None,
        )
        started = time.perf_counter()
        try:
            response = await self._async_client.post(
                self._endpoint,
                json=payload,
                headers=self._headers(),
            )
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(
                f"模型请求超时 ({time.perf_counter() - started:.2f}s): {exc}",
                details={"provider": self._settings.provider},
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelAPIConnectionError(
                f"模型连接失败: {exc}",
                details={"provider": self._settings.provider},
            ) from exc
        return self._parse_response(response, latency=time.perf_counter() - started)

    def _parse_response(self, response: httpx.Response, *, latency: float) -> ChatResponse:
        if response.status_code == 401:
            raise ModelAuthenticationError(
                f"模型认证失败: {response.text[:300]}",
                details={"provider": self._settings.provider, "status_code": response.status_code},
            )
        # 修复：显式拦截 429 限流
        if response.status_code == 429:
            raise ModelRateLimitError(
                f"模型请求被限流 (429): {response.text[:300]}",
                details={"provider": self._settings.provider, "status_code": response.status_code},
            )
        if response.status_code in {400, 200} and "content policy" in response.text.lower():
            raise ModelContentPolicyError(
                f"模型响应被内容策略阻断: {response.text[:300]}",
                details={"provider": self._settings.provider},
            )
        if response.status_code >= 400:
            raise ModelAPIConnectionError(
                f"模型请求失败 ({response.status_code}): {response.text[:500]}",
                details={"provider": self._settings.provider, "status_code": response.status_code},
            )
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ModelPayloadError(
                f"模型返回了无效的 JSON: {response.text[:300]}",
                details={"provider": self._settings.provider},
            ) from exc

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""

        tool_calls = [
            {
                "id": call.get("id") or call.get("tool_call_id"),
                "name": (call.get("function") or {}).get("name"),
                "arguments": _decode_arguments((call.get("function") or {}).get("arguments")),
            }
            for call in (message.get("tool_calls") or [])
            if (call.get("function") or {}).get("name")
        ]
        usage = data.get("usage") or {}
        return ChatResponse(
            content=content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            tool_calls=tool_calls,
            prompt_tokens=_coerce_int(usage.get("prompt_tokens")),
            completion_tokens=_coerce_int(usage.get("completion_tokens")),
            model=(data.get("model") or self._settings.model_name),
            finish_reason=choice.get("finish_reason"),
            provider=self._settings.provider,
            raw={"id": data.get("id"), "latency_seconds": latency},
        )

    def close(self) -> None:
        """关闭同步客户端。"""
        try:
            self._client.close()
        except Exception as exc:
            logger.debug(f"http_client_close_failed | error={exc!s}")

    async def aclose(self) -> None:
        """关闭异步客户端。"""
        try:
            await self._async_client.aclose()
        except Exception as exc:
            logger.debug(f"http_async_client_close_failed | error={exc!s}")

    def __enter__(self) -> _OpenAICompatibleChatModel:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class OpenAICompatibleChatModel(_OpenAICompatibleChatModel):
    """兼容 OpenAI 适配器的公开别名。"""
    name = "openai_compatible"


def _decode_arguments(raw: typing.Any) -> dict[str, typing.Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {"input": raw}
        return decoded if isinstance(decoded, dict) else {"input": decoded}
    return {}


def _coerce_int(value: typing.Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _async_sleep(seconds: float) -> None:
    try:
        import asyncio
        await asyncio.sleep(seconds)
    except RuntimeError:
        time.sleep(seconds)


__all__ = ["OpenAICompatibleChatModel"]

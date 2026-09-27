"""OpenAI-compatible chat model adapter.

This adapter targets endpoints that implement the OpenAI Chat Completions
protocol, including DeepSeek, Qwen/DashScope compatible mode, ZhipuAI's
compatible endpoint, and arbitrary gateways exposing the same contract.
Provider-specific modules may subclass it to adjust defaults or normalize
provider-specific response quirks.

The implementation deliberately uses ``httpx`` directly rather than a heavier
chain wrapper, keeping dependency weight and failure modes explicit.
"""

from __future__ import annotations

import json
import time
import typing

import httpx

from src.model.errors import (
    ModelAPIConnectionError,
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelContentPolicyError,
    ModelPayloadError,
    ModelTimeoutError,
    is_retryable,
)
from src.model.manager import ChatModel, ChatRequest, ChatResponse
from src.observability import get_logger

if typing.TYPE_CHECKING:
    from src.settings import ModelChatSettings

logger = get_logger(__name__)

_DEFAULT_MAX_TOKENS = 2048


class _OpenAICompatibleChatModel(ChatModel):
    """Base adapter for OpenAI-compatible chat endpoints."""

    DEFAULT_BASE_URL = ""
    DEFAULT_MAX_RETRIES = 1

    def __init__(self, settings: ModelChatSettings) -> None:
        self._settings = settings
        self._assert_configured()
        self._client = httpx.Client(timeout=settings.timeout_seconds)

    def _assert_configured(self) -> None:
        if not self._settings.is_configured:
            raise ModelConfigurationError(
                "Chat model API key is empty; configure MODEL_CHAT_API_KEY or settings.yaml.",
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
        payload: dict[str, typing.Any] = {
            "model": self._settings.model_name,
            "messages": request.to_openai_messages(),
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
                logger.warning(
                    "model_retry_scheduled",
                    provider=self._settings.provider,
                    attempt=attempts,
                    delay=delay,
                    error=str(exc),
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
                    "model_retry_scheduled",
                    provider=self._settings.provider,
                    attempt=attempts,
                    delay=delay,
                    error=str(exc),
                )
                await _async_sleep(delay)

    def _post(self, request: ChatRequest) -> ChatResponse:
        payload = self._payload(
            request,
            tools=getattr(self, "_bound_tools", None) or None,
            tool_choice="auto" if getattr(self, "_bound_tools", None) else None,
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
                f"Model request timed out after {time.perf_counter() - started:.2f}s: {exc}",
                details={"provider": self._settings.provider},
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelAPIConnectionError(
                f"Model connection failed: {exc}",
                details={"provider": self._settings.provider},
            ) from exc

        return self._parse_response(response, latency=time.perf_counter() - started)

    async def _apost(self, request: ChatRequest) -> ChatResponse:
        payload = self._payload(
            request,
            tools=getattr(self, "_bound_tools", None) or None,
            tool_choice="auto" if getattr(self, "_bound_tools", None) else None,
        )
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            try:
                response = await client.post(
                    self._endpoint,
                    json=payload,
                    headers=self._headers(),
                )
            except httpx.TimeoutException as exc:
                raise ModelTimeoutError(
                    f"Model request timed out after {time.perf_counter() - started:.2f}s: {exc}",
                    details={"provider": self._settings.provider},
                ) from exc
            except httpx.HTTPError as exc:
                raise ModelAPIConnectionError(
                    f"Model connection failed: {exc}",
                    details={"provider": self._settings.provider},
                ) from exc
        return self._parse_response(response, latency=time.perf_counter() - started)

    def _parse_response(self, response: httpx.Response, *, latency: float) -> ChatResponse:
        if response.status_code == 401:
            raise ModelAuthenticationError(
                f"Model authentication failed: {response.text[:300]}",
                details={"provider": self._settings.provider, "status_code": response.status_code},
            )
        if response.status_code in {400, 200} and "content policy" in response.text.lower():
            raise ModelContentPolicyError(
                f"Model response blocked by content policy: {response.text[:300]}",
                details={"provider": self._settings.provider},
            )
        if response.status_code >= 400:
            raise ModelAPIConnectionError(
                f"Model request failed ({response.status_code}): {response.text[:500]}",
                details={"provider": self._settings.provider, "status_code": response.status_code},
            )
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ModelPayloadError(
                f"Model returned invalid JSON: {response.text[:300]}",
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
        try:
            self._client.close()
        except Exception as exc:
            logger.debug("http_client_close_failed", extra={"error": str(exc)})

    def __enter__(self) -> _OpenAICompatibleChatModel:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class OpenAICompatibleChatModel(_OpenAICompatibleChatModel):
    """Public alias for the OpenAI-compatible adapter."""

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

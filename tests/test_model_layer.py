"""Tests for the pluggable model layer: routing, fallback, and provider errors."""

from __future__ import annotations

from src.model import (
    ChatRequest,
    ChatResponse,
    FallbackExecutor,
    ModelCapabilitySettings,
    ModelRoutingEntry,
    ModelRouter,
    ModelRouterSettings,
    UseCaseBinding,
    classify_model_error,
)
from src.model.errors import (
    ModelAuthenticationError,
    ModelTimeoutError,
)


class _StubModel:
    def __init__(self, name: str, *, fail: bool = False, fail_permanently: bool = False) -> None:
        self.name = name
        self._settings = None
        self.fail = fail
        self.fail_permanently = fail_permanently
        self.calls = 0

    def invoke(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.fail_permanently:
            raise RuntimeError("boom")
        if self.fail:
            raise TimeoutError("timed out")
        return ChatResponse(content=f"{self.name}: {request.messages[-1]['content']}")

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return self.invoke(request)

    def bind_tools(self, tools):
        return self


def _router() -> ModelRouter:
    import os

    os.environ.setdefault("MODEL_CHAT_API_KEY", "test-key")
    return ModelRouter(
        ModelRouterSettings(
            models={
                "chatglm": ModelRoutingEntry(
                    name="chatglm",
                    provider="zhipu",
                    model_name="glm-4-plus",
                    api_key=os.environ["MODEL_CHAT_API_KEY"],
                ),
                "deepseek": ModelRoutingEntry(
                    name="deepseek",
                    provider="deepseek",
                    model_name="deepseek-chat",
                    api_key=os.environ["MODEL_CHAT_API_KEY"],
                ),
                "qwen": ModelRoutingEntry(
                    name="qwen",
                    provider="qwen",
                    model_name="qwen-plus",
                    api_key=os.environ["MODEL_CHAT_API_KEY"],
                ),
            },
            use_cases={
                "chat": UseCaseBinding(
                    use_case="chat", primary="chatglm", fallbacks=["deepseek", "qwen"]
                ),
                "agent": UseCaseBinding(
                    use_case="agent", primary="deepseek", fallbacks=["chatglm"]
                ),
            },
            default_use_case="chat",
        )
    )


def test_router_resolves_chat_chain_order():
    route = _router().route("chat")
    assert route.ordered_names() == ["chatglm", "deepseek", "qwen"]


def test_router_supports_tool_capability():
    router = _router()
    router._configuration.models["chatglm"].capabilities = ModelCapabilitySettings(
        supports_tools=True
    )
    assert router.supports_tools("chatglm") is True
    assert router.supports_tools("missing") is False


def test_executor_falls_back_after_transient_failure():
    primary = _StubModel("primary", fail=True)
    secondary = _StubModel("secondary")
    executor = FallbackExecutor(primary=primary, secondaries=[secondary])

    result = executor.invoke(ChatRequest(messages=[{"role": "user", "content": "hi"}]))

    assert result.succeeded is True
    assert result.response.content == "secondary: hi"
    assert [attempt.provider for attempt in result.attempts] == ["primary", "secondary"]


def test_executor_stops_on_permanent_failure():
    primary = _StubModel("primary", fail_permanently=True)
    secondary = _StubModel("secondary", fail_permanently=True)
    executor = FallbackExecutor(primary=primary, secondaries=[secondary])

    result = executor.invoke(ChatRequest(messages=[{"role": "user", "content": "hi"}]))

    assert result.succeeded is False
    assert result.final_error


def test_classify_timeout_is_retryable():
    classified = classify_model_error(TimeoutError("request timed out"), provider="deepseek")
    assert classified.error_type == "timeout"
    assert classified.retryable is True
    assert isinstance(classified.to_exception(), ModelTimeoutError)


def test_classify_auth_is_not_retryable():
    classified = classify_model_error(Exception("invalid api key"), provider="zhipu")
    assert classified.retryable is False
    assert classified.provider == "zhipu"
    assert isinstance(classified.to_exception(), ModelAuthenticationError)

"""可插拔模型层的测试：路由、Fallback 与提供者异常分类。"""

from __future__ import annotations

from src.model import (
    ChatRequest,
    ChatResponse,
    FallbackExecutor,
    ModelCapabilitySettings,
    ModelRouter,
    ModelRouterSettings,
    ModelRoutingEntry,
    UseCaseBinding,
    classify_model_error,
)
from src.model.errors import (
    ModelAuthenticationError,
    ModelTimeoutError,
)


class _StubModel:
    """最小可用的模型 stub，模拟成功与两类失败。"""

    def __init__(
        self,
        name: str,
        *,
        model_name: str | None = None,
        fail: bool = False,
        fail_permanently: bool = False,
    ) -> None:
        self.name = name
        self.model_name = model_name or name
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
        return ChatResponse(
            content=f"{self.name}: {request.messages[-1]['content']}",
            model=self.model_name,
            provider=self.name,
        )

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return self.invoke(request)

    def bind_tools(self, tools):
        return self


def _build_router(
    *,
    chatglm_capabilities: ModelCapabilitySettings | None = None,
) -> ModelRouter:
    """构造一个固定配置的路由器，用于测试。

    通过参数化 capabilities 来验证 supports_tools，避免直接修改私有字段。
    """
    return ModelRouter(
        ModelRouterSettings(
            models={
                "chatglm": ModelRoutingEntry(
                    name="chatglm",
                    provider="zhipu",
                    model_name="glm-4-plus",
                    api_key="test-key",
                    capabilities=chatglm_capabilities or ModelCapabilitySettings(),
                ),
                "deepseek": ModelRoutingEntry(
                    name="deepseek",
                    provider="deepseek",
                    model_name="deepseek-chat",
                    api_key="test-key",
                ),
                "qwen": ModelRoutingEntry(
                    name="qwen",
                    provider="qwen",
                    model_name="qwen-plus",
                    api_key="test-key",
                ),
            },
            use_cases={
                "chat": UseCaseBinding(
                    use_case="chat",
                    primary="chatglm",
                    fallbacks=["deepseek", "qwen"],
                ),
                "agent": UseCaseBinding(
                    use_case="agent",
                    primary="deepseek",
                    fallbacks=["chatglm"],
                ),
            },
            default_use_case="chat",
        )
    )


def test_router_resolves_chat_chain_order():
    """chat 用例按 primary + fallbacks 顺序解析。"""
    route = _build_router().route("chat")
    assert route.ordered_names() == ["chatglm", "deepseek", "qwen"]


def test_router_supports_tool_capability():
    """capabilities.supports_tools 决定 supports_tools 的返回值。"""
    enabled = _build_router(
        chatglm_capabilities=ModelCapabilitySettings(supports_tools=True)
    )
    disabled = _build_router()

    assert enabled.supports_tools("chatglm") is True
    assert disabled.supports_tools("chatglm") is False
    assert enabled.supports_tools("missing") is False


def test_executor_falls_back_after_transient_failure():
    """首选模型瞬时失败时，切换到备选模型继续执行。"""
    primary = _StubModel("primary", fail=True)
    secondary = _StubModel("secondary")
    executor = FallbackExecutor(primary=primary, secondaries=[secondary])

    result = executor.invoke(
        ChatRequest(messages=[{"role": "user", "content": "hi"}])
    )

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.content == "secondary: hi"
    assert [attempt.provider for attempt in result.attempts] == [
        "primary",
        "secondary",
    ]


def test_executor_stops_on_permanent_failure():
    """不可重试错误触发立即停止，不应继续调用后续模型。"""
    primary = _StubModel("primary", fail_permanently=True)
    secondary = _StubModel("secondary", fail_permanently=True)
    executor = FallbackExecutor(primary=primary, secondaries=[secondary])

    result = executor.invoke(
        ChatRequest(messages=[{"role": "user", "content": "hi"}])
    )

    assert result.succeeded is False
    assert result.final_error
    # RuntimeError("boom") 被归类为不可重试，因此只尝试一次首选模型，
    # 备选模型不应被触碰。
    assert primary.calls == 1
    assert secondary.calls == 0


def test_classify_timeout_is_retryable():
    """超时被归类为可重试的 timeout。"""
    classified = classify_model_error(
        TimeoutError("request timed out"), provider="deepseek"
    )
    assert classified.error_type == "timeout"
    assert classified.retryable is True
    assert isinstance(classified.to_exception(), ModelTimeoutError)


def test_classify_auth_is_not_retryable():
    """认证错误被归类为不可重试的 authentication。"""
    classified = classify_model_error(
        Exception("invalid api key"), provider="zhipu"
    )
    assert classified.retryable is False
    assert classified.provider == "zhipu"
    assert isinstance(classified.to_exception(), ModelAuthenticationError)

"""A/B 测试使用的确定性 Fake 实现。

独立于 tests/_support.py，保证 ab_test 可以单独运行，
不受其他测试 fixture 或 conftest 的影响。
"""

from __future__ import annotations

import typing

from model.protocol import ChatRequest, ChatResponse
from rag.types import RetrievedContext, RetrievedDocument


class FakeChatModel:
    """返回固定答案的确定性模型。"""

    name = "fake"

    def __init__(
        self,
        content: str = "根据[1]《员工手册》，年假需提前5个工作日申请。",
        *,
        provider: str = "fake",
        model_name: str = "fake-model",
    ) -> None:
        self.content = content
        self.provider = provider
        self.model_name = model_name
        self.calls: list[ChatRequest] = []

    def invoke(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse(
            content=self.content,
            model=self.model_name,
            provider=self.provider,
        )

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return self.invoke(request)

    def bind_tools(
        self, tools: typing.Sequence[dict[str, typing.Any]]
    ) -> "FakeChatModel":
        return self


class FakeRetrieval:
    """返回固定证据集的检索器。"""

    def __init__(self, documents: typing.Sequence[RetrievedDocument]) -> None:
        self.documents = list(documents)
        self.calls = 0

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        owner_id: str | None = None,
        allowed_sources: typing.Sequence[str] | None = None,
        **_: typing.Any,
    ) -> RetrievedContext:
        self.calls += 1
        return RetrievedContext(
            query=query,
            documents=list(self.documents),
            search_type="vector",
        )

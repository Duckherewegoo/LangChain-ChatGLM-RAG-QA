"""企业 RAG 测试套件共享的测试替身（Test Doubles）。

本模块刻意不被测试文件通过 ``tests.conftest`` 导入：
项目的 ``tests/`` 目录没有 ``__init__.py``（它不是一个包），
且某些运行环境会暴露一个顶级的 ``tests`` 命名空间掩盖它。
将 Fake 放在一个扁平的模块中可以避免这种解析歧义。
"""

from __future__ import annotations

import typing
from collections.abc import Sequence

from model.protocol import ChatModel, ChatRequest, ChatResponse


class FakeChatModel:
    """用于服务层测试的确定性 Fake 模型。

    模型完全受测试控制：预先填充 ``responses`` 来编写对话脚本，
    在测试后检查 ``tool_calls`` 以断言 Agent 按预期顺序调用了工具。
    它从不请求网络，也从不加载真实的 checkpoint，因此在没有任何
    模型凭证或 GPU 的 CI 环境中使用是安全的。
    """

    name = "fake"

    def __init__(self, settings: typing.Any) -> None:
        self._settings = settings
        self.tool_calls: list[dict[str, typing.Any]] = []
        self.tools: list[dict[str, typing.Any]] = []
        self.responses: list[ChatResponse] = [
            ChatResponse(
                content="基于资料，答案是42。",
                model=getattr(settings, "model_name", "fake-model"),
                provider=self.name,
            )
        ]

    def bind_tools(self, tools: Sequence[dict[str, typing.Any]]) -> "FakeChatModel":
        self.tools = [dict(item) for item in tools]
        return self

    def invoke(self, request: ChatRequest) -> ChatResponse:
        self.tool_calls.append(
            {"messages": list(request.messages), "tools": list(self.tools)}
        )
        if self.responses:
            return self.responses.pop(0)
        return ChatResponse(
            content="完成",
            model=getattr(self._settings, "model_name", "fake-model"),
            provider=self.name,
        )

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return self.invoke(request)


class ToolFirstModel(FakeChatModel):
    """先发起一次工具调用，然后基于检索到的上下文作答的 Fake 模型。"""

    def invoke(self, request: ChatRequest) -> ChatResponse:
        # 第一次调用时，模拟大模型请求调用 retrieve_knowledge
        if not self.tool_calls:
            self.tool_calls.append({"tool": "retrieve_knowledge"})
            return ChatResponse(
                content="",
                model=getattr(self._settings, "model_name", "fake-model"),
                provider=self.name,
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "retrieve_knowledge",
                        "arguments": {"question": request.messages[-1]["content"]},
                    }
                ],
            )
        # 第二次调用时，直接返回最终答案
        return ChatResponse(
            content="根据资料，答案是42。",
            model=getattr(self._settings, "model_name", "fake-model"),
            provider=self.name,
        )


class AuditableModel(FakeChatModel):
    """返回简短、可预测答案的 Fake 模型，用于审计轨迹测试。"""

    def invoke(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            content="审计完成。",
            model=getattr(self._settings, "model_name", "fake-model"),
            provider=self.name,
        )

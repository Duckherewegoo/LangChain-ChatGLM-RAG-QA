"""Shared test doubles for the enterprise-rag-chatglm test suite.

This module is intentionally *not* imported via ``tests.conftest`` from inside
test bodies: the project's ``tests/`` directory has no ``__init__.py`` (it is
not a package) and some runtimes expose a top-level ``tests`` namespace that
shadows it. Keeping fakes in a flat module avoids that resolution ambiguity.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence

from src.model.manager import ChatModel, ChatRequest, ChatResponse


class FakeChatModel(ChatModel):
    """Deterministic fake model used for service tests.

    The model is fully controllable from the test: pre-seed ``responses`` to
    script the conversation, inspect ``tool_calls`` afterwards to assert that
    the agent invoked tools in the expected order. It never contacts a network
    and never loads a real checkpoint, so it is safe to use in CI without any
    model credentials or GPU.
    """

    name = "fake"

    def __init__(self, settings: typing.Any) -> None:
        self._settings = settings
        self.tool_calls: list[dict[str, typing.Any]] = []
        self.tools: list[dict[str, typing.Any]] = []
        self.responses: list[ChatResponse] = [
            ChatResponse(content="基于资料，答案是42。", model=settings.model_name)
        ]

    def bind_tools(self, tools: Sequence[dict[str, typing.Any]]) -> ChatModel:
        self.tools = [dict(item) for item in tools]
        return self

    def invoke(self, request: ChatRequest) -> ChatResponse:
        self.tool_calls.append(
            {"messages": list(request.messages), "tools": list(self.tools)}
        )
        if self.responses:
            return self.responses.pop(0)
        return ChatResponse(content="完成", model=self._settings.model_name)

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return self.invoke(request)


class ToolFirstModel(FakeChatModel):
    """Fake model that issues one tool call, then answers from retrieved context."""

    def invoke(self, request: ChatRequest) -> ChatResponse:  # noqa: D401 - test double
        if not self.tool_calls:
            self.tool_calls.append({"tool": "retrieve_knowledge"})
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "retrieve_knowledge",
                        "arguments": {"question": request.messages[-1]["content"]},
                    }
                ],
            )
        return ChatResponse(content="根据资料，答案是42。", model=self._settings.model_name)


class AuditableModel(FakeChatModel):
    """Fake model that returns a short, predictable answer for audit tests."""

    def invoke(self, request: ChatRequest) -> ChatResponse:  # noqa: D401 - test double
        return ChatResponse(content="审计完成。", model=self._settings.model_name)

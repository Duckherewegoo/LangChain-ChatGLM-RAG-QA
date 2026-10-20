"""Chat model abstraction and normalized payloads.

定义全局使用的 ChatModel 协议接口与标准化的请求、响应数据模型。

业务层只能依赖本模块，具体的厂商实现位于 model.providers，
通过 model.registry 注册。业务代码绝对不可直接引用具体厂商的 SDK。
"""

from __future__ import annotations

import typing
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """与厂商无关的对话补全请求。"""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_langchain_messages(self) -> list[BaseMessage]:
        """将存储的字典消息转换为 LangChain 消息对象。"""
        output: list[BaseMessage] = []
        for message in self.messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                output.append(SystemMessage(content=content))
            elif role == "user":
                output.append(HumanMessage(content=content))
            elif role == "assistant":
                kwargs: dict[str, Any] = {"content": content}
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    kwargs["tool_calls"] = tool_calls
                output.append(AIMessage(**kwargs))
            elif role == "tool":
                output.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=message.get("tool_call_id", ""),
                    )
                )
            else:
                output.append(HumanMessage(content=content))
        return output


class ChatResponse(BaseModel):
    """标准化的大模型响应模型。"""

    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    finish_reason: str | None = None
    provider: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    def has_tool_calls(self) -> bool:
        """当模型请求了至少一个工具调用时返回 True。"""
        return bool(self.tool_calls)


@typing.runtime_checkable
class ChatModel(typing.Protocol):
    """
    大模型协议接口。

    具体厂商必须实现以下方法。使用 typing.Protocol 实现结构化子类型，
    这意味着实现类不需要显式继承本类，只要方法签名一致即可。
    """

    name: str

    def invoke(self, request: ChatRequest) -> ChatResponse:
        """执行同步对话补全。"""
        ...

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        """执行异步对话补全。"""
        ...

    def bind_tools(self, tools: Sequence[dict[str, Any]]) -> "ChatModel":
        """返回一个配置了指定工具的模型实例。"""
        ...


__all__ = [
    "ChatModel",
    "ChatRequest",
    "ChatResponse",
]

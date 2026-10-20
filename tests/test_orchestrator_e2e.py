"""多角色编排器的端到端测试（harness + agent）。

这些测试使用确定性的 fake 模型和 fake 检索，但会实际执行 orchestrator、
reviewer 和审计轨迹代码路径 —— 不对内部方法做 mock patch。
意图是验证 retrieve -> reason -> review 流水线能正确组合。
"""

from __future__ import annotations

import typing

import pytest

from exceptions import ValidationError
from model.fallback import FallbackExecutor
from model.protocol import ChatRequest, ChatResponse
from rag.orchestrator import AgentOrchestrator, OrchestratorRequest
from rag.reviewer import AnswerReviewer, ReviewReport
from rag.tools import ToolRegistry
from rag.types import RetrievedContext, RetrievedDocument
from settings import AgentSettings, GuardrailsSettings, ReviewSettings


class _FakeModelSettings:
    """最小化的 ModelChatSettings 替身。"""

    model_name = "fake-model"
    provider = "fake"


class _FakeModel:
    """返回固定带引用答案的确定性模型。

    不继承 ChatModel Protocol，通过鸭子类型满足协议要求。
    """

    name = "fake"

    def __init__(self, settings: typing.Any | None = None) -> None:
        self._settings = settings or _FakeModelSettings()
        self.calls: list[ChatRequest] = []

    def invoke(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse(
            content="根据[1]《员工手册》，年假需提前5个工作日申请。",
            model=self._settings.model_name,
            provider=self._settings.provider,
        )

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return self.invoke(request)

    def bind_tools(
        self, tools: typing.Sequence[dict[str, typing.Any]]
    ) -> "_FakeModel":
        return self


class _FakeReviewer(AnswerReviewer):
    """永远通过的 Reviewer，用于验证流水线能走完。"""

    def __init__(self) -> None:
        # AnswerReviewer 需要 model 参数，这里传 None 表示不用 LLM 审核；
        # review 方法会被覆写，所以内部不会真正调用模型。
        super().__init__(model=None)

    def review(  # type: ignore[override]
        self,
        question: str,
        answer: str,
        evidence: typing.Sequence[dict[str, typing.Any]],
    ) -> ReviewReport:
        return ReviewReport(
            approved=True, issues=[], summary="approved", strategy="fake"
        )


class _FakeRetrieval:
    """返回单条证据文档的确定性检索器。"""

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        owner_id: str | None = None,
        allowed_sources: typing.Sequence[str] | None = None,
        **_: typing.Any,
    ) -> RetrievedContext:
        documents = [
            RetrievedDocument(
                document_id="doc1",
                chunk_id="doc1#0",
                content="员工手册：年假需提前5个工作日申请。",
                source="manual.pdf",
                title="员工手册",
                score=0.9,
                tenant_id=tenant_id,
            )
        ]
        return RetrievedContext(
            query=query, documents=documents, search_type="vector"
        )


def _build_orchestrator(
    *,
    model: typing.Any | None = None,
    retrieval: typing.Any | None = None,
    reviewer: typing.Any | None = None,
    review_enabled: bool = True,
) -> AgentOrchestrator:
    """构造一个固定依赖的编排器，让测试可以参数化配置。

    用参数化替代"直接改 orchestrator._agent_settings"的私有属性写法。
    """
    m = model or _FakeModel()
    executor = FallbackExecutor(
        primary=m, secondaries=[], max_attempts_per_provider=1
    )
    return AgentOrchestrator(
        executor=executor,
        retrieval=retrieval or _FakeRetrieval(),  # type: ignore[arg-type]
        reviewer=reviewer or _FakeReviewer(),  # type: ignore[arg-type]
        registry=ToolRegistry(),
        agent_settings=AgentSettings(
            review=ReviewSettings(enabled=review_enabled, use_llm=False)
        ),
        guardrails_settings=GuardrailsSettings(),
    )


def test_orchestrator_runs_retrieve_reason_review() -> None:
    """完整流水线：检索 -> 推理 -> 审核 -> 输出。"""
    orchestrator = _build_orchestrator()
    response = orchestrator.run(
        OrchestratorRequest(question="如何申请年假？", tenant_id="tenant-a")
    )

    assert response.answer
    assert response.review_status == "approved"
    assert response.review is not None
    assert response.review.approved is True
    # 三个角色都必须被调用
    assert {"retriever", "reasoner", "reviewer"} <= set(response.roles_invoked)
    assert response.review_rounds >= 0

    # 审计轨迹必须包含真实事件
    summary = response.trail.summarize()
    assert summary["event_count"] > 0
    assert summary["duration_seconds"] >= 0
    assert summary["models"] or summary["providers"] or summary["tools"]


def test_orchestrator_review_disabled_skips_review() -> None:
    """审核禁用时仍然产出答案，但审核策略标记为 disabled。"""
    orchestrator = _build_orchestrator(review_enabled=False)
    response = orchestrator.run(
        OrchestratorRequest(question="年假几天？", tenant_id="tenant-a")
    )

    assert response.answer
    assert response.review is not None
    assert response.review.strategy == "disabled"
    # 禁用审核时不应调用 reviewer 角色
    assert "reviewer" not in response.roles_invoked


def test_orchestrator_rejects_empty_question() -> None:
    """空问题在 guardrails 层被拦截。"""
    orchestrator = _build_orchestrator()
    with pytest.raises(ValidationError):
        orchestrator.run(
            OrchestratorRequest(question="   ", tenant_id="tenant-a")
        )


def test_orchestrator_returns_safe_refusal_when_no_evidence() -> None:
    """检索为空时，编排器必须返回安全拒答，不得凭空回答。"""

    class _EmptyRetrieval(_FakeRetrieval):
        def retrieve(
            self,
            query: str,
            **_: typing.Any,
        ) -> RetrievedContext:
            return RetrievedContext(
                query=query, documents=[], search_type="vector"
            )

    orchestrator = _build_orchestrator(retrieval=_EmptyRetrieval())
    response = orchestrator.run(
        OrchestratorRequest(question="机密问题", tenant_id="tenant-a")
    )

    assert response.answer
    # 安全拒答文案必须包含"没有"或"无法"其中之一
    assert "没有" in response.answer or "无法" in response.answer

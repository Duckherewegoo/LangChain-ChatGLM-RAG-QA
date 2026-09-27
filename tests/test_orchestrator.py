"""End-to-end tests for the multi-role orchestrator (harness + agent).

These tests use real, deterministic fakes for the model and vector store but
exercise the actual orchestrator, reviewer, and audit-trail code paths -- no
mock patching of internal methods. The intent is to verify that the
retrieve -> reason -> review pipeline composes correctly.
"""

from __future__ import annotations

import typing

import pytest

from src.model.manager import ChatModel, ChatRequest, ChatResponse
from src.model.fallback import FallbackExecutor
from src.rag.orchestrator import OrchestratorRequest, AgentOrchestrator
from src.rag.reviewer import AnswerReviewer
from src.rag.retrieval import RetrievalPipeline, RetrievedContext, RetrievedDocument
from src.rag.tools import Guardrails, ToolRegistry
from src.settings import AgentSettings, GuardrailsSettings, ReviewSettings


class _FakeModelSettings:
    """Minimal stand-in for ModelChatSettings."""

    model_name = "fake-model"
    provider = "fake"


class _FakeModel(ChatModel):
    """Returns a scripted, citation-bearing answer."""

    name = "fake"

    def __init__(self, settings: typing.Any | None = None) -> None:
        self._settings = settings or _FakeModelSettings()
        self.calls: list[ChatRequest] = []

    def invoke(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        # ``provider`` may be absent on minimal stand-ins (e.g. ``type("S", ...)``).
        provider = getattr(self._settings, "provider", None)
        return ChatResponse(
            content="根据[1]《员工手册》，年假需提前5个工作日申请。",
            model=self._settings.model_name,
            provider=provider,
        )

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return self.invoke(request)

    def bind_tools(self, tools: typing.Sequence[dict[str, typing.Any]]) -> ChatModel:
        return self


class _FakeReviewer(AnswerReviewer):
    """Reviewer that always approves -- used to verify the pipeline completes."""

    def review(self, question, answer, evidence):
        from src.rag.reviewer import ReviewReport

        return ReviewReport(approved=True, issues=[], summary="approved", strategy="fake")


class _FakeRetrieval:
    """Stand-in retrieval pipeline returning one evidence document."""

    def retrieve(self, query, *, tenant_id="default", owner_id=None, allowed_sources=None, **_):
        from src.rag.types import RetrievedContext

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
            query=query,
            documents=documents,
            search_type="vector",
        )


@pytest.fixture
def orchestrator() -> AgentOrchestrator:
    model = _FakeModel(type("S", (), {"model_name": "fake"})())
    executor = FallbackExecutor(primary=model, secondaries=[], max_attempts_per_provider=1)
    retrieval = _FakeRetrieval()  # type: ignore[assignment]
    reviewer = _FakeReviewer()
    return AgentOrchestrator(
        executor=executor,
        retrieval=retrieval,  # type: ignore[arg-type]
        reviewer=reviewer,
        registry=ToolRegistry(),
        agent_settings=AgentSettings(review=ReviewSettings(enabled=True, use_llm=False)),
        guardrails_settings=GuardrailsSettings(),
    )


def test_orchestrator_runs_retrieve_reason_review(orchestrator):
    request = OrchestratorRequest(
        question="如何申请年假？",
        tenant_id="tenant-a",
    )
    response = orchestrator.run(request)

    assert response.answer, "orchestrator must produce an answer"
    assert response.review_status == "approved"
    assert response.review is not None
    assert response.review.approved is True
    assert "retriever" in response.roles_invoked
    assert "reasoner" in response.roles_invoked
    assert "reviewer" in response.roles_invoked
    assert response.review_rounds >= 0
    # Audit trail must be populated with real events
    summary = response.trail.summarize()
    assert summary["event_count"] > 0
    assert summary["duration_seconds"] >= 0
    # At least one of models/providers/tools must be recorded
    assert summary["models"] or summary["providers"] or summary["tools"]


def test_orchestrator_review_disabled_skips_review(orchestrator):
    orchestrator._agent_settings = AgentSettings(review=ReviewSettings(enabled=False))
    request = OrchestratorRequest(question="年假几天？", tenant_id="tenant-a")
    response = orchestrator.run(request)

    # When review is disabled, the orchestrator must still produce an answer.
    assert response.answer, "orchestrator must produce an answer even when review is disabled"
    assert response.review is not None
    # The report strategy must reflect that review was disabled/skipped.
    assert response.review.strategy in ("disabled", "skipped")


def test_orchestrator_rejects_empty_question(orchestrator):
    from src.exceptions import ValidationError

    request = OrchestratorRequest(question="   ", tenant_id="tenant-a")
    with pytest.raises(ValidationError):
        orchestrator.run(request)


def test_orchestrator_returns_safe_refusal_when_no_evidence():
    """When retrieval yields nothing, the orchestrator must not invent."""

    class _EmptyRetrieval(_FakeRetrieval):
        def retrieve(self, *a, **kw):
            return RetrievalContext(query=a[0], documents=[], search_type="vector")

    model = _FakeModel(type("S", (), {"model_name": "fake"})())
    executor = FallbackExecutor(primary=model, secondaries=[], max_attempts_per_provider=1)
    orchestrator = AgentOrchestrator(
        executor=executor,
        retrieval=_EmptyRetrieval(),  # type: ignore[arg-type]
        reviewer=_FakeReviewer(),
        registry=ToolRegistry(),
        agent_settings=AgentSettings(review=ReviewSettings(enabled=True, use_llm=False)),
        guardrails_settings=GuardrailsSettings(),
    )
    response = orchestrator.run(OrchestratorRequest(question="机密问题", tenant_id="tenant-a"))
    assert response.answer
    assert "没有" in response.answer or "无法" in response.answer

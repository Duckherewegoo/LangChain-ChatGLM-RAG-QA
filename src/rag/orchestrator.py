"""多角色协作编排器（Multi-Agent Orchestrator）。

职责边界：
    - 编排：retrieve -> reason -> review 循环 -> finalize
    - 审计：每一步写入 AgentAuditTrail
    - 组装：把内部 Blackboard 映射为对外 OrchestratorResponse
    检索实现、Prompt 细节、审核策略、模型选择、工具调用均不在此模块。

审核循环语义：
    AgentSettings.review.max_retries = N 表示「最多重试 N 次」，
    因此审核最多执行 N + 1 轮（首次审核 + N 次带反馈重试）。
    reasoner 主动拒答（无可用证据）时只审核一次，不再重试。

失败策略（故意不对称）：
    - 检索失败：软失败，记录错误后继续，交由 reasoner 走 can_handle=False；
    - 推理失败：硬失败，抛 AgentExecutionError 中断流水线。
"""

from __future__ import annotations

import time
import typing
import uuid

from pydantic import BaseModel, Field

from base.exceptions import AgentExecutionError, ValidationError
from model.protocol import ChatRequest, ChatResponse
from base.observability import get_logger
from rag.agent_events import AgentAuditTrail, AgentEventType
from rag.reviewer import AnswerReviewer, ReviewReport
from rag.roles import (
    AgentDefinition,
    Blackboard,
    Role,
    enterprise_qa_definition,
)
from rag.tool_sources import LocalToolSource
from rag.tools import Guardrails, ToolRegistry
from settings import AgentSettings, GuardrailsSettings

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from model.fallback import FallbackExecutor
    from rag.retrieval import RetrievalPipeline
    from rag.types import RetrievedDocument

logger = get_logger(__name__)


class OrchestratorRequest(BaseModel):
    """对外的编排请求。"""

    question: str
    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class OrchestratorResponse(BaseModel):
    """编排结果。"""

    request_id: str
    answer: str
    citations: list[dict[str, typing.Any]] = Field(default_factory=list)
    review: ReviewReport | None = None
    review_status: str = "not_applicable"
    review_rounds: int = 0
    roles_invoked: list[str] = Field(default_factory=list)
    documents: list[dict[str, typing.Any]] = Field(default_factory=list)
    trail: AgentAuditTrail
    duration_seconds: float = 0.0
    model: str | None = None

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "request_id": self.request_id,
            "answer": self.answer,
            "citations": self.citations,
            "review_status": self.review_status,
            "review": self.review.to_dict() if self.review else None,
            "review_rounds": self.review_rounds,
            "roles_invoked": self.roles_invoked,
            "documents": self.documents,
            "trail": self.trail.summarize(),
            "duration_seconds": self.duration_seconds,
            "model": self.model,
        }


class AgentOrchestrator:
    """把角色协作编排成可运行的流水线。"""

    def __init__(
        self,
        executor: FallbackExecutor,
        retrieval: RetrievalPipeline,
        reviewer: AnswerReviewer,
        *,
        registry: ToolRegistry | None = None,
        agent_settings: AgentSettings | None = None,
        guardrails_settings: GuardrailsSettings | None = None,
        definition: AgentDefinition | None = None,
    ) -> None:
        self._executor = executor
        self._retrieval = retrieval
        self._reviewer = reviewer
        self._agent_settings = agent_settings or AgentSettings()
        self._guardrails_settings = guardrails_settings or GuardrailsSettings()
        self._guardrails = Guardrails(
            self._guardrails_settings, agent_settings=self._agent_settings
        )
        self._definition = definition or enterprise_qa_definition()

        self._roles: dict[str, Role] = {
            role.definition.name: role for role in self._definition.roles
        }
        self._registry = registry or ToolRegistry(
            sources=[LocalToolSource()], allowed_tools=[]
        )

    def run(self, request: OrchestratorRequest) -> OrchestratorResponse:
        blackboard = self._init_blackboard(request)
        trail = blackboard.metadata["trail"]
        start_time = time.perf_counter()

        trail.record_now(
            event_type=AgentEventType.STARTED,
            node="orchestrator",
            data={
                "definition": self._definition.name,
                "roles": self._definition.role_names(),
                "tenant_id": request.tenant_id,
            },
        )

        try:
            self._guardrails.check_question(request.question)
        except ValidationError:
            raise
        # 仅 wrap 未预期异常；校验错误保持原始类型向上抛
        except Exception as exc:
            raise ValidationError(f"输入被拒绝: {exc}") from exc

        try:
            self._run_pipeline(request, blackboard, trail)
        except AgentExecutionError:
            raise
        except ValidationError:
            raise
        except Exception as exc:
            logger.exception(f"编排流水线失败 | request_id={request.request_id}")
            raise AgentExecutionError(
                f"编排执行失败: {exc}",
                details={"request_id": request.request_id},
            ) from exc

        response = self._build_response(
            request, blackboard, trail, time.perf_counter() - start_time
        )
        trail.complete()
        return response

    def _run_pipeline(
        self,
        request: OrchestratorRequest,
        blackboard: Blackboard,
        trail: AgentAuditTrail,
    ) -> None:
        """依次执行 retrieve -> reason -> review 循环 -> finalize。"""
        self._invoke_retriever(request, blackboard, trail)
        self._invoke_reasoner(request, blackboard, trail)

        review_config = self._agent_settings.review
        if not review_config.enabled:
            blackboard.review = ReviewReport(
                approved=True, summary="审核已禁用。", strategy="disabled"
            )
            blackboard.metadata["review_status"] = "skipped"
            trail.record_now(
                event_type=AgentEventType.REVIEW_STARTED,
                node="reviewer",
                status="skipped",
                message="根据配置，审核环节已被禁用。",
            )
            return

        # 首次审核 + 最多 max_retries 次带反馈重试
        max_retries = max(0, review_config.max_retries)
        for review_attempt in range(max_retries + 1):
            blackboard.review_rounds = review_attempt
            report = self._invoke_reviewer(blackboard, trail)
            blackboard.review = report

            if report.approved or not report.retryable():
                break

            # reasoner 已明确拒答（无可用证据）时，仅审一次，不重试
            if blackboard.metadata.get("reasoner_declined"):
                break

            trail.record_now(
                event_type=AgentEventType.REVIEW_OVERRIDDEN,
                node="reviewer",
                status="retry",
                data={"round": review_attempt + 1, "issues": report.blocking},
            )
            self._invoke_reasoner(
                request, blackboard, trail, feedback=report.summary
            )

    def _invoke_retriever(
        self,
        request: OrchestratorRequest,
        blackboard: Blackboard,
        trail: AgentAuditTrail,
    ) -> None:
        role = self._roles["retriever"]
        if not role.can_handle(blackboard):
            return
        blackboard.roles_invoked.append(role.definition.name)
        trail.record_now(
            event_type=AgentEventType.ROLE_DISPATCHED,
            node="retriever",
            data={"purpose": role.definition.purpose},
        )

        try:
            retrieval_result = self._retrieval.retrieve(
                blackboard.original_question,
                tenant_id=request.tenant_id,
                owner_id=request.owner_id,
                allowed_sources=request.allowed_sources,
            )
        except Exception as exc:
            logger.warning(f"检索失败 | error={exc!s}")
            blackboard.record_error("retrieve", str(exc))
            trail.record_now(
                event_type=AgentEventType.TOOL_CALL_FAILED,
                node="retriever",
                status="error",
                message=f"检索失败：{exc}",
                retryable=False,
            )
            return

        blackboard.evidence = [
            self._document_to_evidence(document)
            for document in retrieval_result.documents
        ]
        trail.record_now(
            event_type=AgentEventType.RETRIEVAL,
            node="retriever",
            data={
                "document_count": len(retrieval_result.documents),
                "search_type": retrieval_result.search_type,
            },
        )

    def _invoke_reasoner(
        self,
        request: OrchestratorRequest,
        blackboard: Blackboard,
        trail: AgentAuditTrail,
        feedback: str | None = None,
    ) -> None:
        role = self._roles["reasoner"]
        if not role.can_handle(blackboard):
            blackboard.answer_draft = "（当前没有可用于回答的授权证据。）"
            blackboard.roles_invoked.append(role.definition.name)
            blackboard.metadata["reasoner_declined"] = True
            trail.record_now(
                event_type=AgentEventType.ROLE_DISPATCHED,
                node="reasoner",
                status="skipped",
                message="无可用授权证据，reasoner 直接拒答。",
            )
            return

        blackboard.roles_invoked.append(role.definition.name)
        trail.record_now(
            event_type=AgentEventType.ROLE_DISPATCHED,
            node="reasoner",
            data={
                "evidence_count": len(blackboard.evidence),
                "feedback": bool(feedback),
            },
        )

        prompt = role.build_prompt(blackboard)
        if feedback:
            prompt += "\n\n审核反馈（请据此修正，不要引用不存在的来源）：\n" + feedback

        model_response = self._call_model(prompt, role=role, trail=trail)
        blackboard.answer_draft = (model_response.content or "").strip() or None
        blackboard.metadata["reasoning_model"] = model_response.model

    def _invoke_reviewer(
        self,
        blackboard: Blackboard,
        trail: AgentAuditTrail,
    ) -> ReviewReport:
        role = self._roles["reviewer"]
        blackboard.roles_invoked.append(role.definition.name)
        trail.record_now(event_type=AgentEventType.REVIEW_STARTED, node="reviewer")

        report = self._reviewer.review(
            blackboard.original_question,
            blackboard.answer_draft or "",
            blackboard.evidence,
        )

        trail.record_now(
            event_type=AgentEventType.REVIEW_COMPLETED,
            node="reviewer",
            status="approved" if report.approved else "blocking",
            data={
                "strategy": report.strategy,
                "issue_count": len(report.issues),
                "blocking": len(report.blocking),
                "model": report.model,
            },
        )
        for issue in report.issues:
            trail.record_now(
                event_type=AgentEventType.REVIEW_ISSUE,
                node="reviewer",
                message=issue.get("message", ""),
                status=issue.severity,
                data={
                    "code": issue.code,
                    "suggestion": issue.suggestion,
                },
            )
        return report

    def _call_model(
        self,
        prompt: str,
        *,
        role: Role,
        trail: AgentAuditTrail,
    ) -> ChatResponse:
        """通过 FallbackExecutor 调用模型，记录每次尝试。"""
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]
        request = ChatRequest(messages=messages, tools=[])

        execution_result = self._executor.invoke(request)

        for attempt in execution_result.attempts:
            trail.record_now(
                event_type=AgentEventType.MODEL_CALL,
                node=role.definition.name,
                model=attempt.model_name,
                provider=attempt.provider,
                status="ok" if attempt.status == "success" else "error",
                message=attempt.error_message if attempt.status != "success" else None,
                retryable=attempt.retryable,
                latency_seconds=attempt.elapsed_seconds,
            )

        if not execution_result.succeeded or not execution_result.response:
            raise AgentExecutionError(
                f"执行角色 {role.definition.name!r} 时所有模型均调用失败。",
                details={
                    "role": role.definition.name,
                    "attempts": [
                        {"provider": a.provider, "error": a.error_message}
                        for a in execution_result.attempts
                    ],
                },
            )

        return execution_result.response

    def _system_prompt(self) -> str:
        """拼接产品契约与协作规则，作为系统提示前缀。

        角色列表由各 Role.build_prompt 自行负责，此处不重复描述。
        """
        definition = self._definition
        return (
            f"你是 {definition.name}（{definition.version}），{definition.product_purpose}\n"
            f"输入契约：{definition.input_contract}\n"
            f"输出契约：{definition.output_contract}\n"
            "协作规则：先检索、再推理、最后审核；证据不足时明确说明无法确认，不得编造。"
        )

    def _init_blackboard(self, request: OrchestratorRequest) -> Blackboard:
        trail = AgentAuditTrail(request_id=request.request_id)
        board = Blackboard(
            request_id=request.request_id,
            original_question=request.question,
            tenant_id=request.tenant_id,
        )
        board.metadata["trail"] = trail
        return board

    def _build_response(
        self,
        request: OrchestratorRequest,
        blackboard: Blackboard,
        trail: AgentAuditTrail,
        elapsed_seconds: float,
    ) -> OrchestratorResponse:
        report = blackboard.review

        # 优先使用流水线阶段写入的状态（如审核禁用/跳过时为 "skipped"）
        pipeline_status = blackboard.metadata.get("review_status")
        if pipeline_status:
            review_status = str(pipeline_status)
        elif report is None:
            review_status = "not_applicable"
        elif not report.approved and report.blocking:
            review_status = "rejected"
        elif report.warnings:
            review_status = "warning"
        else:
            review_status = "approved"

        if review_status == "rejected":
            # 不暴露被拒草稿
            answer = _refusal_message(blackboard)
        else:
            answer = blackboard.answer_draft or _refusal_message(blackboard)

        response = OrchestratorResponse(
            request_id=request.request_id,
            answer=answer,
            citations=_build_citations(blackboard.evidence),
            review=report,
            review_status=review_status,
            review_rounds=blackboard.review_rounds,
            roles_invoked=list(blackboard.roles_invoked),
            documents=blackboard.evidence,
            trail=trail,
            duration_seconds=elapsed_seconds,
            model=blackboard.metadata.get("reasoning_model"),
        )
        trail.record_now(
            event_type=AgentEventType.FINALIZED,
            node="orchestrator",
            status=review_status,
            data={"answer_replaced": review_status == "rejected"},
        )
        return response

    @staticmethod
    def _document_to_evidence(document: RetrievedDocument) -> dict[str, typing.Any]:
        return {
            "document_id": document.document_id,
            "chunk_id": document.chunk_id,
            "source": document.source,
            "title": document.title,
            "page": document.page,
            "score": document.score,
            "rerank_score": document.rerank_score,
            "tenant_id": document.tenant_id,
            "owner_id": document.owner_id,
            "snippet": _snippet(document.content),
            "content": document.content,
        }


def _refusal_message(blackboard: Blackboard) -> str:
    """生成拒答文案，按「无证据 / 有证据但审核拒绝」两种原因区分。"""
    if not blackboard.evidence:
        return (
            "当前没有可访问的授权文档能够回答该问题。"
            "请确认文档已入库，或联系知识库管理员。"
        )
    if blackboard.review is not None and not blackboard.review.approved:
        return (
            "已检索到相关文档，但回答未通过审核，暂不提供该结论。"
            "建议补充或缩小问题范围后重试。"
        )
    return (
        "根据已授权知识库，无法找到足以回答该问题的证据。"
        "建议补充相关文档或缩小问题范围。"
    )


def _snippet(text: str, max_length: int = 600) -> str:
    if not isinstance(text, str):
        text = str(text)
    normalized = " ".join(text.split())
    return normalized[:max_length] + ("…" if len(normalized) > max_length else "")


def _build_citations(
    evidence: list[dict[str, typing.Any]]
) -> list[dict[str, typing.Any]]:
    return [
        {
            "index": index + 1,
            "source": item.get("source"),
            "title": item.get("title"),
            "page": item.get("page"),
            "score": item.get("score"),
        }
        for index, item in enumerate(evidence)
    ]


__all__ = [
    "AgentOrchestrator",
    "OrchestratorRequest",
    "OrchestratorResponse",
]

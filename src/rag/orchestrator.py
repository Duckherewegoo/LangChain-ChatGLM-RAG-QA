"""多角色协作编排器（Multi-Agent Orchestrator）。

本项目由多个职责单一的角色协作完成企业知识库问答。
本模块把这些角色编排成一张可观测的流水线：
    retrieve -> reason -> review -> finalize
                            |
                            +--（审核不通过，retryable）--> reason
"""

from __future__ import annotations

import time
import typing
import uuid

from pydantic import BaseModel, Field

from exceptions import AgentExecutionError, PermissionDeniedError, ValidationError
from model.protocol import ChatRequest, ChatResponse
from observability import get_logger
from rag.agent_events import AgentAuditTrail, AgentEventType
from rag.reviewer import AnswerReviewer, ReviewReport
from rag.roles import (
    AgentDefinition,
    Blackboard,
    Role,
    enterprise_qa_definition,
)
from rag.tool_sources import LocalToolSource
from rag.tool_types import ToolContext
from rag.tools import Guardrails, ToolCallRecord, ToolRegistry
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
    history: list[dict[str, str]] = Field(default_factory=list)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    use_hybrid: bool = True


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
        started = time.perf_counter()

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

        trail.complete()
        return self._build_response(request, blackboard, trail, time.perf_counter() - started)

    def _run_pipeline(
        self,
        request: OrchestratorRequest,
        blackboard: Blackboard,
        trail: AgentAuditTrail,
    ) -> None:
        """依次执行 retrieve -> reason -> review 循环 -> finalize。"""
        self._invoke_retriever(request, blackboard, trail)
        self._invoke_reasoner(request, blackboard, trail)

        review_cfg = self._agent_settings.review
        if not review_cfg.enabled:
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

        max_rounds = max(0, review_cfg.max_retries)
        for _round in range(max_rounds + 1):
            blackboard.review_rounds = _round
            report = self._invoke_reviewer(blackboard, trail)
            blackboard.review = report

            if report.approved or not report.retryable():
                break

            trail.record_now(
                event_type=AgentEventType.REVIEW_OVERRIDDEN,
                node="reviewer",
                status="retry",
                data={"round": _round + 1, "issues": report.blocking},
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
        blackboard.roles_invoked.append(role.definition.name)
        trail.record_now(
            event_type=AgentEventType.ROLE_DISPATCHED,
            node="retriever",
            data={"purpose": role.definition.purpose},
        )

        try:
            context = self._retrieval.retrieve(
                blackboard.original_question,
                tenant_id=request.tenant_id,
                owner_id=request.owner_id,
                allowed_sources=list(request.allowed_sources),
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
            self._document_to_evidence(document) for document in context.documents
        ]
        blackboard.rewritten_queries = [blackboard.original_question]
        trail.record_now(
            event_type=AgentEventType.RETRIEVAL,
            node="retriever",
            data={
                "document_count": len(context.documents),
                "search_type": context.search_type,
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

        response = self._call_model(prompt, role=role, trail=trail)
        blackboard.answer_draft = (response.content or "").strip() or None
        blackboard.metadata["reasoning_model"] = response.model

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
                status=issue.get("severity", "warning"),
                message=issue.get("message", ""),
                data={
                    "code": issue.get("code"),
                    "suggestion": issue.get("suggestion"),
                },
            )
        return report

    def invoke_tool(
        self,
        name: str,
        arguments: dict[str, typing.Any],
        *,
        request_id: str,
        tenant_id: str,
        owner_id: str | None = None,
        allowed_sources: list[str] | None = None,
    ) -> ToolCallRecord:
        """在白名单内调用一个工具，全程可审计。

        注意：当前流水线不直接调用工具，此方法预留给外部显式调用。
        """
        context = ToolContext(
            tenant_id=tenant_id,
            owner_id=owner_id,
            allowed_sources=list(allowed_sources or []),
            request_id=request_id,
        )
        self._guardrails.validate_tool_call(name, self._registry)

        try:
            handler = self._registry.get(name)
            result = handler(arguments or {}, context)
            return ToolCallRecord(tool=name, arguments=arguments, result=result)
        except ValidationError:
            raise
        except PermissionDeniedError:
            raise
        except Exception as exc:
            logger.warning(
                f"工具调用失败 | tool={name} | request_id={request_id} | error={exc!s}"
            )
            return ToolCallRecord(
                tool=name, arguments=arguments, status="error", error=str(exc)
            )

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

        result = self._executor.invoke(request)

        for attempt in result.attempts:
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

        if not result.succeeded or not result.response:
            raise AgentExecutionError(
                f"执行角色 {role.definition.name!r} 时所有模型均调用失败。",
                details={
                    "role": role.definition.name,
                    "attempts": [
                        {"provider": a.provider, "error": a.error_message}
                        for a in result.attempts
                    ],
                },
            )

        return result.response

    def _system_prompt(self) -> str:
        """拼接角色契约与产品约束，作为系统提示前缀。"""
        definition = self._definition
        roles_text = "\n".join(
            f"- {role.definition.title}（{role.definition.name}）：{role.definition.purpose}"
            for role in definition.roles
        )
        return (
            f"你是 {definition.name}（{definition.version}），{definition.product_purpose}\n"
            f"输入契约：{definition.input_contract}\n"
            f"输出契约：{definition.output_contract}\n"
            f"参与角色：\n{roles_text}\n"
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
        duration: float,
    ) -> OrchestratorResponse:
        report = blackboard.review

        # 优先使用流水线写入的 review_status（例如 disabled 时为 "skipped"）
        stored_status = blackboard.metadata.get("review_status")
        if stored_status:
            status = str(stored_status)
        elif report is None:
            status = "not_applicable"
        elif not report.approved and report.blocking:
            status = "rejected"
        elif report.warnings:
            status = "warning"
        else:
            status = "approved"

        # 审核被拒时返回安全拒答，不把被拒的草稿暴露给用户
        if status == "rejected":
            answer = _safe_refusal(blackboard)
        else:
            answer = blackboard.answer_draft or _safe_refusal(blackboard)

        return OrchestratorResponse(
            request_id=request.request_id,
            answer=answer,
            citations=_evidence_to_citations(blackboard.evidence),
            review=report if isinstance(report, ReviewReport) else None,
            review_status=status,
            review_rounds=blackboard.review_rounds,
            roles_invoked=list(dict.fromkeys(blackboard.roles_invoked)),
            documents=blackboard.evidence,
            trail=trail,
            duration_seconds=duration,
            model=blackboard.metadata.get("reasoning_model"),
        )

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


def _safe_refusal(blackboard: Blackboard) -> str:
    """当没有任何可用答案时返回的安全兜底文案。"""
    if blackboard.evidence:
        return "根据已授权知识库，无法找到足以回答该问题的证据。建议补充相关文档或缩小问题范围。"
    return "当前没有可访问的授权文档能够回答该问题。请确认文档已入库，或联系知识库管理员。"


def _snippet(text: str, limit: int = 600) -> str:
    if not isinstance(text, str):
        text = str(text)
    cleaned = " ".join(text.split())
    return cleaned[:limit] + ("…" if len(cleaned) > limit else "")


def _evidence_to_citations(
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

"""多角色协作编排器（Multi-Agent Orchestrator）。

本项目只有一条用户故事——"企业知识库问答"——但它由多个**职责单一**的角色协作
完成。本模块就是把这些角色编排成一张可观测的 LangGraph 状态机：

    retrieve  ->  reason  ->  review  ->  finalize
                              |
                              +--（审核不通过，retryable）--> reason

设计要点
--------
* **单进程、多角色**，不是多进程多 Agent。每个角色是一个数据对象
  （:class:`~src.rag.roles.Role`）+ 一段提示词，没有单独的服务、没有
  RPC。这符合"就业导向实战项目"的定位：核心是**职责分离与协作模式**，
  不是分布式系统的复杂度。
* **共享黑板** :class:`Blackboard` 是角色间唯一的通信媒介。角色只读自己
  关心的键，写自己负责的键。这比强类型通道更易审计、更易扩展。
* **审核是硬闸**，不是建议。审核返回 blocking issue 时，按配置重试推理；
  超过重试次数则返回安全拒答，并在响应里标记 ``review_status=rejected``。
* **可降级**。LLM 调用、工具调用、审核全部可能失败，每一步都有 try/except
  兜底：模型不可用 -> 走 FallbackExecutor；工具失败 -> 记录错误并继续；
  审核失败 -> 降级为规则审核或放行（按策略）。
* **可观测**。每一步都产生 :class:`~src.rag.agent_events.AgentEventType`
  事件，最终响应包含完整轨迹，可直接用 ``/agent/{id}/trail`` 复盘。
"""

from __future__ import annotations

import logging
import time
import typing
import uuid

from pydantic import BaseModel, Field

from src.exceptions import AgentExecutionError, PermissionDeniedError, ValidationError
from src.model.manager import ChatRequest, ChatResponse
from src.rag.agent_events import AgentAuditTrail, AgentEventType
from src.rag.reviewer import AnswerReviewer, ReviewReport
from src.rag.roles import (
    AgentDefinition,
    Blackboard,
    Role,
    enterprise_qa_definition,
)
from src.rag.tool_sources import LocalToolSource
from src.rag.tools import Guardrails, ToolCallRecord, ToolContext, ToolRegistry
from src.settings import AgentSettings, GuardrailsSettings

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.model.fallback import FallbackExecutor
    from src.rag.retrieval import RetrievalPipeline
    from src.rag.types import RetrievedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公共请求 / 响应
# ---------------------------------------------------------------------------


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
    review_status: str = "not_applicable"  # "approved" | "rejected" | "warning" | "skipped"
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


# ---------------------------------------------------------------------------
# 编排器
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """把角色协作编排成可运行的流水线。

    它不继承某个"超级 Agent"，也不持有任何模型的具体引用——所有模型访问
    都通过 :class:`ChatModel` 接口与 :class:`~src.model.fallback.FallbackExecutor`
    完成。这是一个**协调者**，不是上帝对象。
    """

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
        # 默认注册表：如果没有注入，就用一个空的内存工具源（单角色测试用）
        self._registry = registry or ToolRegistry(
            sources=[LocalToolSource()], allowed_tools=[]
        )

    # ---- 公共入口 ----------------------------------------------------------

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
            raise ValidationError(f"Input rejected: {exc}") from exc

        try:
            self._run_pipeline(request, blackboard, trail)
        except AgentExecutionError:
            raise
        except ValidationError:
            raise
        except Exception as exc:
            logger.exception("orchestrator_pipeline_failed", extra={"request_id": request.request_id})
            raise AgentExecutionError(
                f"Orchestration failed: {exc}",
                details={"request_id": request.request_id},
            ) from exc

        trail.complete()
        return self._build_response(request, blackboard, trail, time.perf_counter() - started)

    # ---- 流水线 ------------------------------------------------------------

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
            blackboard.review = ReviewReport(approved=True, summary="审核已禁用。", strategy="disabled")
            blackboard.metadata["review_status"] = "skipped"
            trail.record_now(
                event_type=AgentEventType.REVIEW_STARTED,
                node="reviewer",
                status="skipped",
                message="Review disabled by configuration.",
            )
            return

        max_rounds = max(0, review_cfg.max_retries)
        for _round in range(max_rounds + 1):
            blackboard.review_rounds = _round
            report = self._invoke_reviewer(request, blackboard, trail)
            blackboard.review = report

            if report.approved or not report.retryable():
                break

            trail.record_now(
                event_type=AgentEventType.REVIEW_OVERRIDDEN,
                node="reviewer",
                status="retry",
                data={"round": _round + 1, "issues": report.blocking},
            )
            self._invoke_reasoner(request, blackboard, trail, feedback=report.summary)

    # -- 角色执行：retriever ------------------------------------------------

    def _invoke_retriever(
        self,
        request: OrchestratorRequest,
        blackboard: Blackboard,
        trail: AgentAuditTrail,
    ) -> None:
        role = self._roles["retriever"]
        blackboard.roles_invoked.append(role.definition.name)
        trail.record_now(event_type=AgentEventType.ROLE_DISPATCHED, node="retriever", data={"purpose": role.definition.purpose})

        try:
            context = self._retrieval.retrieve(
                blackboard.original_question,
                tenant_id=request.tenant_id,
                owner_id=request.owner_id,
                allowed_sources=list(request.allowed_sources),
            )
        except Exception as exc:
            logger.warning("retriever_failed", extra={"error": str(exc)})
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
            data={"document_count": len(context.documents), "search_type": context.search_type},
        )

    # -- 角色执行：reasoner -------------------------------------------------

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
            data={"evidence_count": len(blackboard.evidence), "feedback": bool(feedback)},
        )

        prompt = role.build_prompt(blackboard)
        if feedback:
            prompt += (
                "\n\n审核反馈（请据此修正，不要引用不存在的来源）：\n" + feedback
            )

        response = self._call_model(prompt, role=role, trail=trail)
        blackboard.answer_draft = (response.content or "").strip() or None
        blackboard.metadata["reasoning_model"] = response.model

    # -- 角色执行：reviewer -------------------------------------------------

    def _invoke_reviewer(
        self,
        request: OrchestratorRequest,
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
                data={"code": issue.get("code"), "suggestion": issue.get("suggestion")},
            )
        return report

    # -- 工具调用（供 reasoner 触发的受控工具） -----------------------------

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
        """在白名单内调用一个工具，全程可审计。"""

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
            logger.warning("tool_failed", extra={"tool": name, "request_id": request_id, "error": str(exc)})
            return ToolCallRecord(
                tool=name, arguments=arguments, status="error", error=str(exc)
            )

    # ---- 内部工具 ----------------------------------------------------------

    def _call_model(
        self,
        prompt: str,
        *,
        role: Role,
        trail: AgentAuditTrail,
    ) -> ChatResponse:
        """通过 FallbackExecutor 调用模型，记录每次尝试。"""

        messages = [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": prompt}]
        request = ChatRequest(messages=messages, tools=[])

        for attempt_index, model in enumerate(self._executor._chain):  # noqa: SLF001 - introspection for observability
            started = time.perf_counter()
            try:
                response = model.invoke(request)
            except Exception as exc:
                logger.warning(
                    "model_call_failed",
                    extra={
                        "role": role.definition.name,
                        "provider": getattr(model, "name", None),
                        "error": str(exc),
                        "exc_type": type(exc).__name__,
                    },
                )
                trail.record_now(
                    event_type=AgentEventType.MODEL_CALL,
                    node=role.definition.name,
                    provider=getattr(model, "name", None),
                    status="error",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                )
                continue

            trail.record_now(
                event_type=AgentEventType.MODEL_CALL,
                node=role.definition.name,
                model=getattr(response, "model", None) or getattr(model, "name", None),
                provider=getattr(model, "name", None),
                status="ok",
                latency_seconds=time.perf_counter() - started,
            )
            if attempt_index > 0:
                trail.record_now(
                    event_type=AgentEventType.MODEL_FALLBACK,
                    node=role.definition.name,
                    provider=getattr(model, "name", None),
                    status="ok",
                    data={"attempt": attempt_index + 1},
                )
            return response

        raise AgentExecutionError(
            f"All models failed while executing role {role.definition.name!r}.",
            details={"role": role.definition.name, "request_id": None},
        )

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
        if report is None:
            status = "not_applicable"
        elif not report.approved and report.blocking:
            status = "rejected"
        elif report.warnings:
            status = "warning"
        else:
            status = "approved"

        answer = blackboard.answer_draft or _safe_refusal(blackboard)

        return OrchestratorResponse(
            request_id=request.request_id,
            answer=answer,
            citations=_evidence_to_citations(blackboard.evidence),
            review=report,
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


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


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


def _evidence_to_citations(evidence: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]:
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

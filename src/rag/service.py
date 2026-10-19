"""协调检索、提示词构建和生成的高层 RAG 服务。

本模块负责“检索优先”的答案构建。模型选择与回退执行委托给 FallbackExecutor；
本模块绝不直接导入提供者 SDK，也不会静默捕获模型异常。当执行器发生回退时，
它会记录参与的模型，以便审计响应。
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field

from src.exceptions import ModelInvocationError, RetrievalError, ValidationError
from src.model import ChatModel, ChatRequest, ChatResponse, FallbackExecutor, FallbackResult
from src.observability import get_logger

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from src.model.embedding import EmbeddingModel
    from src.rag.agent_events import AgentAuditTrail
    from src.rag.retrieval import RetrievalPipeline
    from src.rag.types import RetrievedContext
    from src.settings import GuardrailsSettings, RetrievalSettings

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你是企业级知识库智能助手。仅基于提供的上下文回答用户问题，保持专业、准确、简洁。
若上下文不足以回答，请明确说明无法确认，不得臆测。引用上下文编号回答；如涉及数字、日期或具体操作，必须可追溯。"""


class AnswerRequest(BaseModel):
    """终端用户的回答请求。"""

    question: str = Field(..., min_length=1)
    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    top_k: int | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    model: str | None = None
    use_case: str = "chat"


class Citation(BaseModel):
    """生成的答案所引用的来源。"""

    index: int
    source: str
    title: str = ""
    page: int | None = None
    chunk_id: str = ""


class AnswerResult(BaseModel):
    """最终答案及其证据与执行元数据。"""

    answer: str
    query: str
    context: RetrievedContext
    citations: list[Citation] = Field(default_factory=list)
    context_used: bool = False
    model: str | None = None
    tokens: dict[str, int | None] = Field(default_factory=dict)
    attempted_models: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class ModelExecutionFailed(ModelInvocationError):
    """当回退链中的所有模型均失败时抛出。"""
    status_code = 502
    error_code = "model_execution_failed"


class RAGService:
    """生产级检索增强生成服务。"""

    def __init__(
        self,
        retrieval: RetrievalPipeline,
        chat_model: ChatModel,
        executor: FallbackExecutor,
        embedder: EmbeddingModel,
        settings: RetrievalSettings,
        guardrails: GuardrailsSettings,
    ) -> None:
        self._retrieval = retrieval
        self._chat_model = chat_model
        self._executor = executor
        self._embedder = embedder
        self._settings = settings
        self._guardrails = guardrails

    def answer(self, request: AnswerRequest) -> AnswerResult:
        """执行一次检索优先的直接回答。"""
        self._validate_request(request)
        context = self._retrieve(request)
        prompt_messages = self._build_prompt(request.question, context, request.history)

        result = self._executor.invoke(ChatRequest(messages=prompt_messages))
        response = self._ensure_success(result, request.use_case)

        return self._build_answer_result(request, context, response, result)

    async def aanswer(self, request: AnswerRequest) -> AnswerResult:
        """异步执行一次检索优先的直接回答。"""
        self._validate_request(request)
        context = self._retrieve(request)
        prompt_messages = self._build_prompt(request.question, context, request.history)

        result = await self._executor.ainvoke(ChatRequest(messages=prompt_messages))
        response = self._ensure_success(result, request.use_case)

        return self._build_answer_result(request, context, response, result)

    def retrieve_only(self, request: AnswerRequest) -> RetrievedContext:
        """仅执行检索，不调用模型生成。"""
        self._validate_request(request)
        return self._retrieve(request)

    def _retrieve(self, request: AnswerRequest) -> RetrievedContext:
        try:
            return self._retrieval.retrieve(
                request.question,
                tenant_id=request.tenant_id,
                top_k=request.top_k,
                allowed_sources=request.allowed_sources,
                owner_id=request.owner_id,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"知识检索失败: {exc}") from exc

    def _ensure_success(self, result: FallbackResult, use_case: str) -> ChatResponse:
        """校验执行结果，如果全部失败则抛出异常。"""
        if result.succeeded and result.response is not None:
            return result.response

        raise ModelExecutionFailed(
            "所有配置的对话模型均无法生成响应。",
            details={
                "attempts": [attempt.model_dump() for attempt in result.attempts],
                "use_case": use_case,
            },
        )

    def _build_answer_result(
        self,
        request: AnswerRequest,
        context: RetrievedContext,
        response: ChatResponse,
        result: FallbackResult,
    ) -> AnswerResult:
        """统一构建最终响应对象。"""
        # 正确判断是否触发回退：成功的尝试不是第一个，或者尝试次数大于1且发生了失败
        fallback_used = len(result.attempts) > 1

        return AnswerResult(
            answer=response.content,
            query=request.question,
            context=context,
            citations=self._build_citations(context),
            context_used=not context.is_empty(),
            model=response.model,
            tokens={
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
            attempted_models=self._executor.providers,
            fallback_used=fallback_used,
        )

    def _build_prompt(
        self, question: str, context: RetrievedContext, history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        history_limit = max(0, self._guardrails.max_history_messages)

        for entry in history[-history_limit:]:
            role = entry.get("role")
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": entry.get("content", "")})

        if not context.is_empty():
            messages.append(
                {
                    "role": "system",
                    "content": f"以下为受权限控制的检索上下文，编号对应最终引用：\n{_format_context(context)}",
                }
            )
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _build_citations(context: RetrievedContext) -> list[Citation]:
        return [
            Citation(
                index=index,
                source=document.source,
                title=document.title,
                page=document.page,
                chunk_id=document.chunk_id,
            )
            for index, document in enumerate(context.documents, start=1)
        ]

    def _validate_request(self, request: AnswerRequest) -> None:
        question = (request.question or "").strip()
        if not question:
            raise ValidationError("问题内容不能为空。")
        limit = getattr(self._guardrails, "max_question_length", 0) or 2000
        if len(question) > limit:
            raise ValidationError(f"问题长度超出了 {limit} 字符的最大限制。")


def _format_context(context: RetrievedContext) -> str:
    if context.is_empty():
        return "（当前问题没有检索到可引用的知识库内容）"
    blocks: list[str] = []
    for index, document in enumerate(context.documents, start=1):
        location = f"第{document.page}页" if isinstance(document.page, int) and document.page > 0 else ""
        header = f"[{index}] {document.title or document.source}{(' ' + location).rstrip()}"
        blocks.append(f"{header}\n{document.content}")
    return "\n\n".join(blocks)


def build_answer_result_from_agent(
    answer: str,
    query: str,
    context: RetrievedContext,
    response: ChatResponse | None,
    attempted_models: list[str],
    fallback_used: bool,
    trail: AgentAuditTrail,
) -> AnswerResult:
    """从智能体执行结果构建 AnswerResult。"""
    return AnswerResult(
        answer=answer,
        query=query,
        context=context,
        citations=[
            Citation(
                index=index,
                source=document.source,
                title=document.title,
                page=document.page,
                chunk_id=document.chunk_id,
            )
            for index, document in enumerate(context.documents, start=1)
        ],
        context_used=not context.is_empty(),
        model=response.model if response else None,
        attempted_models=attempted_models,
        fallback_used=fallback_used,
    )


__all__ = [
    "AnswerRequest",
    "AnswerResult",
    "Citation",
    "ModelExecutionFailed",
    "RAGService",
    "build_answer_result_from_agent",
]

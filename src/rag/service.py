"""High-level RAG service coordinating retrieval, prompt building, and generation.

This module owns retrieval-first answer construction. Model selection and
fallback execution are delegated to :class:`FallbackExecutor`; this module never
imports provider SDKs or catches model exceptions silently. When the executor
fails over, it records which models participated so the response remains
auditable.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field

from src.exceptions import RetrievalError, ValidationError
from src.model import ChatModel, ChatRequest, ChatResponse, FallbackExecutor
from src.observability import get_logger

if typing.TYPE_CHECKING:
    from src.model.embedding import EmbeddingModel
    from src.rag.agent_events import AgentAuditTrail
    from src.rag.retrieval import RetrievalPipeline
    from src.rag.types import RetrievedContext
    from src.settings import GuardrailsSettings, RetrievalSettings

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你是企业级知识库智能助手。仅基于提供的上下文回答用户问题，保持专业、准确、简洁。
若上下文不足以回答，请明确说明无法确认，不得臆测。引用上下文编号回答；如涉及数字、日期或具体操作，必须可追溯。"""


class AnswerRequest(BaseModel):
    """End-user answer request."""

    question: str = Field(..., min_length=1)
    tenant_id: str = "default"
    owner_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    top_k: int | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    model: str | None = None
    use_case: str = "chat"


class Citation(BaseModel):
    """A source referenced by the generated answer."""

    index: int
    source: str
    title: str = ""
    page: int | None = None
    chunk_id: str = ""


class AnswerResult(BaseModel):
    """Final answer together with evidence and execution metadata."""

    answer: str
    query: str
    context: RetrievedContext
    citations: list[Citation] = Field(default_factory=list)
    context_used: bool = False
    model: str | None = None
    tokens: dict[str, int | None] = Field(default_factory=dict)
    attempted_models: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class RAGService:
    """Production-style retrieval-augmented generation service."""

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
        """Run a retrieval-first direct answer."""

        self._validate_request(request)
        context = self._retrieve(request)
        prompt_messages = self._build_prompt(request.question, context, request.history)
        response = self._generate(ChatRequest(messages=prompt_messages), request)
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
            fallback_used=self._was_fallback(response),
        )

    async def aanswer(self, request: AnswerRequest) -> AnswerResult:
        self._validate_request(request)
        context = self._retrieve(request)
        prompt_messages = self._build_prompt(request.question, context, request.history)
        response = await self._generate_async(ChatRequest(messages=prompt_messages), request)
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
            fallback_used=self._was_fallback(response),
        )

    def retrieve_only(self, request: AnswerRequest) -> RetrievedContext:
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
            raise RetrievalError(f"Knowledge retrieval failed: {exc}") from exc

    def _generate(self, request: ChatRequest, original: AnswerRequest) -> ChatResponse:
        result = self._executor.invoke(request)
        if result.succeeded and result.response is not None:
            return result.response
        raise ModelExecutionFailed(
            "All configured chat models failed to generate a response.",
            details={
                "attempts": [attempt.model_dump() for attempt in result.attempts],
                "use_case": original.use_case,
            },
        )

    async def _generate_async(self, request: ChatRequest, original: AnswerRequest) -> ChatResponse:
        result = await self._executor.ainvoke(request)
        if result.succeeded and result.response is not None:
            return result.response
        raise ModelExecutionFailed(
            "All configured chat models failed to generate a response.",
            details={
                "attempts": [attempt.model_dump() for attempt in result.attempts],
                "use_case": original.use_case,
            },
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
        citations: list[Citation] = []
        for index, document in enumerate(context.documents, start=1):
            citations.append(
                Citation(
                    index=index,
                    source=document.source,
                    title=document.title,
                    page=document.page,
                    chunk_id=document.chunk_id,
                )
            )
        return citations

    def _validate_request(self, request: AnswerRequest) -> None:
        question = (request.question or "").strip()
        if not question:
            raise ValidationError("Question must not be empty.")
        limit = getattr(self._guardrails, "max_question_length", 0) or 2000
        if len(question) > limit:
            raise ValidationError(f"Question exceeds maximum length of {limit} characters.")

    @staticmethod
    def _was_fallback(response: ChatResponse) -> bool:
        return bool(response.provider)


def _format_context(context: RetrievedContext) -> str:
    if context.is_empty():
        return "（当前问题没有检索到可引用的知识库内容）"
    blocks: list[str] = []
    for index, document in enumerate(context.documents, start=1):
        location = f"第{document.page}页" if isinstance(document.page, int) and document.page > 0 else ""
        header = f"[{index}] {document.title or document.source}{(' ' + location).rstrip()}"
        blocks.append(f"{header}\n{document.content}")
    return "\n\n".join(blocks)


class ModelExecutionFailed(RetrievalError):
    """Raised when every model in the fallback chain fails."""

    status_code = 502
    error_code = "model_execution_failed"


def build_answer_result_from_agent(
    answer: str,
    query: str,
    context: RetrievedContext,
    response: ChatResponse | None,
    attempted_models: list[str],
    fallback_used: bool,
    trail: AgentAuditTrail,
) -> AnswerResult:
    """Create an :class:`AnswerResult` from an agent execution outcome."""

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

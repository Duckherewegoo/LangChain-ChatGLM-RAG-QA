"""检索策略层：混合检索编排器。"""

from __future__ import annotations

import typing

from pydantic import Field

from observability import get_logger
from rag.types import RetrievedContext

from .fusers import ResultFuser
from .retrievers import RetrievalStrategy
from .rewriters import QueryRewriter
from .types import Candidate, QueryRewrite

logger = get_logger(__name__)

# RetrievalContext.candidates 里 document.content 的截断长度（字符数）。
# 该字段仅用于审计 / 调试；最终对外 documents 仍使用完整原文。
_CANDIDATE_CONTENT_LIMIT = 500


class RetrievalContext(RetrievedContext):
    """带策略审计信息的检索上下文。"""
    rewrite: QueryRewrite = Field(default_factory=lambda: QueryRewrite(original="", rewritten=""))
    candidates: list[Candidate] = Field(default_factory=list)
    strategy_notes: list[dict[str, typing.Any]] = Field(default_factory=list)

    def distinct_sources(self) -> list[str]:
        seen: list[str] = []
        for document in self.documents:
            if document.source not in seen:
                seen.append(document.source)
        return seen


class HybridRetrieval:
    """编排查询改写、多路召回与结果融合。"""

    def __init__(
        self, rewriter: QueryRewriter, strategies: typing.Sequence[tuple[str, RetrievalStrategy]],
        fuser: ResultFuser, *, per_query_top_k: int = 20,
    ) -> None:
        if not strategies:
            raise ValueError("HybridRetrieval 至少需要一条检索通道。")
        self._rewriter = rewriter
        self._strategies = list(strategies)
        self._fuser = fuser
        self._per_query_top_k = max(1, int(per_query_top_k))

    @property
    def rewriter(self) -> QueryRewriter:
        return self._rewriter

    @property
    def fuser(self) -> ResultFuser:
        return self._fuser

    def retrieve(
        self, question: str, *, tenant_id: str = "default", top_k: int = 8,
        allowed_sources: typing.Sequence[str] | None = None, owner_id: str | None = None,
    ) -> RetrievalContext:
        """执行完整链路：改写 -> 多路召回 -> 融合。"""
        rewrite = self._rewriter.rewrite(question)
        # all_queries() 只调一次：若返回生成器，第二次调用会拿到空序列。
        queries = list(rewrite.all_queries())
        all_candidates: list[Candidate] = []
        per_query: list[dict[str, typing.Any]] = []

        for query in queries:
            for label, strategy in self._strategies:
                candidates = strategy.retrieve(
                    query, tenant_id=tenant_id, top_k=self._per_query_top_k,
                    allowed_sources=allowed_sources, owner_id=owner_id,
                )
                per_query.append({"query": query, "strategy": label, "candidate_count": len(candidates)})
                all_candidates.extend(candidates)

        fused = self._fuser.fuse(all_candidates, top_k=top_k)
        strategy_labels = [label for label, _ in self._strategies]
        context = RetrievalContext(
            query=question,
            documents=[entry.document for entry in fused],
            search_type="hybrid" if len(strategy_labels) > 1 else strategy_labels[0],
            rewrite=rewrite,
            candidates=_snippet_candidates(all_candidates),
            strategy_notes=per_query,
        )
        logger.info(
            "混合检索完成 | 改写方法=%s | 查询数=%d | 策略=%s | 融合结果=%d",
            rewrite.method,
            len(queries),
            strategy_labels,
            len(fused),
        )
        return context


def _snippet_candidates(
    candidates: list[Candidate], *, max_chars: int = _CANDIDATE_CONTENT_LIMIT
) -> list[Candidate]:
    """复制候选，把 document.content 截断到 max_chars。

    仅作用于审计字段 RetrievalContext.candidates；由 fuser 产出的
    documents 使用完整原文。截断后的 Candidate 是新的 frozen 实例，
    原 Candidate 与其 document 不受影响。
    """
    truncated: list[Candidate] = []
    for candidate in candidates:
        content = candidate.document.content or ""
        if len(content) <= max_chars:
            truncated.append(candidate)
            continue
        document = candidate.document.model_copy(
            update={"content": content[:max_chars] + "…"}
        )
        truncated.append(
            Candidate(
                document=document,
                score=candidate.score,
                source=candidate.source,
                rank=candidate.rank,
            )
        )
    return truncated

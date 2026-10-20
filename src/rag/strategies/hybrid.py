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
        all_candidates: list[Candidate] = []
        per_query: list[dict[str, typing.Any]] = []

        for query in rewrite.all_queries():
            for label, strategy in self._strategies:
                candidates = strategy.retrieve(
                    query, tenant_id=tenant_id, top_k=self._per_query_top_k,
                    allowed_sources=allowed_sources, owner_id=owner_id,
                )
                per_query.append({"query": query, "strategy": label, "candidate_count": len(candidates)})
                all_candidates.extend(candidates)

        fused = self._fuser.fuse(all_candidates, top_k=top_k)
        context = RetrievalContext(
            query=question,
            documents=[entry.document for entry in fused],
            search_type="hybrid" if len(self._strategies) > 1 else self._strategies[0][0],
            rewrite=rewrite, candidates=all_candidates, strategy_notes=per_query,
        )
        logger.info(
            f"混合检索完成 | 改写方法={rewrite.method} | 查询数={len(rewrite.all_queries())} | "
            f"策略={[label for label, _ in self._strategies]} | 融合结果={len(fused)}"
        )
        return context

"""检索管线：向量检索、权限过滤与重排序。"""

from __future__ import annotations

import typing
from contextlib import nullcontext

from pydantic import BaseModel

from exceptions import RetrievalError
from model.embedding import EmbeddingModel
from observability import get_logger, get_metric
from rag.reranker import Reranker, create_reranker
from rag.types import RetrievedContext, RetrievedDocument

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from rag.vector_store import VectorStore
    from settings import RetrievalSettings

logger = get_logger(__name__)


class RetrievedEvidence(BaseModel):
    """在检索与答案构建之间传递的内部证据数据对象。"""

    document: RetrievedDocument
    vector_score: float
    rerank_score: float | None = None


class RetrievalPipeline:
    """将向量检索与重排协调为带类型的检索上下文。"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
        settings: RetrievalSettings,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._settings = settings
        self._reranker = reranker or create_reranker(
            settings.rerank.enabled, settings.rerank.model_name
        )

    @property
    def settings(self) -> RetrievalSettings:
        return self._settings

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        top_k: int | None = None,
        allowed_sources: typing.Sequence[str] | None = None,
        owner_id: str | None = None,
    ) -> RetrievedContext:
        """返回授权且重排后的检索证据。"""
        if not query.strip():
            raise ValueError("检索查询不能为空。")

        limit = max(1, top_k or self._settings.top_k)
        fetch_k = limit * max(1, self._settings.fetch_multiplier)
        query_embedding = self._embedder.embed_query(query)

        try:
            candidates = self._store.search(
                query_embedding,
                tenant_id=tenant_id,
                top_k=fetch_k,
                allowed_sources=allowed_sources,
            )
        except Exception as exc:
            raise RetrievalError(f"向量检索失败: {exc}") from exc

        evidence = [
            RetrievedEvidence(
                document=self._to_document(candidate.record, candidate.score),
                vector_score=candidate.score,
            )
            for candidate in candidates
            if self._is_authorized(candidate.record, owner_id)
        ]

        retrieval_documents = get_metric("retrieval_documents")
        if retrieval_documents is not None:
            retrieval_documents.labels(tenant_id=tenant_id).observe(len(evidence))

        if not evidence:
            logger.info(f"检索为空 | query_len={len(query)} | tenant_id={tenant_id}")
            return RetrievedContext(query=query, documents=[], search_type="vector")

        if self._settings.rerank.enabled:
            evidence = self._apply_reranking(query, evidence)

        documents = self._to_context_documents(evidence[:limit])
        context = RetrievedContext(query=query, documents=documents, search_type="vector")
        logger.info(
            f"检索完成 | tenant_id={tenant_id} | candidates={fetch_k} | "
            f"retrieved={len(documents)} | reranked={self._settings.rerank.enabled}"
        )
        return context

    def _apply_reranking(
        self, query: str, evidence: list[RetrievedEvidence]
    ) -> list[RetrievedEvidence]:
        """应用交叉编码器重排，并合并原始向量分数。"""
        rerank_latency = get_metric("rerank_latency")
        context = rerank_latency.time() if rerank_latency else nullcontext()
        with context:
            results = self._reranker.rerank(
                query, [entry.document for entry in evidence], top_k=len(evidence)
            )

        evidence_map = {entry.document.chunk_id: entry for entry in evidence}
        reranked: list[RetrievedEvidence] = []

        for result in results:
            entry = evidence_map.get(result.document.chunk_id)
            if entry:
                reranked.append(
                    RetrievedEvidence(
                        document=entry.document,
                        vector_score=entry.vector_score,
                        rerank_score=result.score,
                    )
                )

        reranked.sort(
            key=lambda item: (
                item.rerank_score
                if item.rerank_score is not None
                else item.vector_score
            ),
            reverse=True,
        )
        return reranked

    def _to_context_documents(
        self, evidence: typing.Sequence[RetrievedEvidence]
    ) -> list[RetrievedDocument]:
        """将内部证据转换为对外的检索文档列表。"""
        documents: list[RetrievedDocument] = []
        for entry in evidence:
            documents.append(
                entry.document.model_copy(update={"rerank_score": entry.rerank_score})
            )
        return documents

    @staticmethod
    def _is_authorized(record: typing.Any, owner_id: str | None) -> bool:
        """判断文档是否对当前主体可见。

        一旦指定 owner_id，只返回 owner_id 严格匹配的文档；
        None 不视为通配。未指定 owner_id 时不做归属过滤（由上游负责租户隔离）。
        """
        if owner_id is None:
            return True
        return getattr(record, "owner_id", None) == owner_id

    @staticmethod
    def _to_document(record: typing.Any, score: float) -> RetrievedDocument:
        return RetrievedDocument(
            document_id=record.document_id,
            chunk_id=record.chunk_id,
            title=record.title,
            source=record.source,
            content=record.text,
            score=score,
            tenant_id=record.tenant_id,
            owner_id=record.owner_id,
            tags=list(record.tags or []),
            metadata=dict(record.metadata or {}),
        )


__all__ = ["RetrievalPipeline", "RetrievedEvidence"]

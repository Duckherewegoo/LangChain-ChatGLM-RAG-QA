"""检索管线：向量检索、权限过滤与重排序。"""

from __future__ import annotations

import typing
from contextlib import nullcontext

from pydantic import BaseModel

from exceptions import RetrievalError, ValidationError
from model.embedding import EmbeddingModel
from observability import get_logger, get_metric
from rag.reranker import Reranker, create_reranker
from rag.types import RetrievedContext, RetrievedDocument

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from rag.vector_store import VectorRecord, VectorStore
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
        search_type: str = "vector",
    ) -> RetrievedContext:
        """返回授权且重排后的检索证据。

        search_type 由上层声明（如 hybrid 层传 "hybrid"）；默认 "vector"，
        保持本类对调用方无感知。
        """
        query = query.strip()
        if not query:
            raise ValidationError("检索查询不能为空。")

        # top_k 显式 None 用默认；<=0 也回退到默认，避免下游 0 或负数导致空结果。
        limit = top_k if top_k is not None and top_k > 0 else self._settings.top_k
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
            # 去掉 tenant_id label：租户数高时会造成 Prometheus series 爆炸。
            # 租户维度应放在 trace/log，而非 metric。
            retrieval_documents.observe(len(evidence))

        if not evidence:
            logger.info(
                "检索为空 | query_len=%d | tenant_id=%s", len(query), tenant_id
            )
            return RetrievedContext(
                query=query, documents=[], search_type=search_type
            )

        if self._settings.rerank.enabled:
            evidence = self._apply_reranking(query, evidence)

        documents = self._to_context_documents(evidence[:limit])
        context = RetrievedContext(
            query=query, documents=documents, search_type=search_type
        )
        logger.info(
            "检索完成 | tenant_id=%s | candidates=%d | retrieved=%d | reranked=%s",
            tenant_id,
            fetch_k,
            len(documents),
            self._settings.rerank.enabled,
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
            if entry is None:
                # reranker 返回了证据集中不存在的 chunk_id，可能是模型幻觉或
                # ID 映射错误；记录 warning 而不是静默丢弃。
                logger.warning(
                    "reranker 返回未知 chunk_id | chunk_id=%s",
                    result.document.chunk_id,
                )
                continue
            # 仅采用 rerank 分数；document 以原始检索结果为准（reranker 可能
            # 返回 enrich 过的 document，但这里不采用其元数据变更）。
            reranked.append(
                RetrievedEvidence(
                    document=entry.document,
                    vector_score=entry.vector_score,
                    rerank_score=result.score,
                )
            )

        # 保守排序：有 rerank_score 的按 rerank_score 降序排在前；
        # 没有 rerank_score 的按 vector_score 降序排在后。
        # 不混合两种量纲的分数直接比较。
        reranked.sort(
            key=lambda item: (
                item.rerank_score is None,
                -(item.rerank_score or 0.0),
            )
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
    def _is_authorized(record: "VectorRecord", owner_id: str | None) -> bool:
        """判断文档是否对当前主体可见。

        一旦指定 owner_id，只返回 owner_id 严格匹配的文档；
        None 不视为通配。未指定 owner_id 时不做归属过滤（由上游负责租户隔离）。
        """
        if owner_id is None:
            return True
        return getattr(record, "owner_id", None) == owner_id

    @staticmethod
    def _to_document(record: "VectorRecord", score: float) -> RetrievedDocument:
        # record.text 在部分 vector store 实现中可能为 None（只存向量不存原文），
        # 此时用空串兜底，避免 RetrievedDocument.content（str）校验失败。
        return RetrievedDocument(
            document_id=record.document_id,
            chunk_id=record.chunk_id,
            title=record.title,
            source=record.source,
            content=record.text or "",
            score=score,
            page=record.page,
            tenant_id=record.tenant_id,
            owner_id=record.owner_id,
            tags=list(record.tags or []),
            metadata=dict(record.metadata or {}),
        )


__all__ = ["RetrievalPipeline", "RetrievedEvidence"]

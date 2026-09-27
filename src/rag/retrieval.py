"""Retrieval pipeline: vector search, authorization filtering, and reranking."""

from __future__ import annotations

import typing

from pydantic import BaseModel

from src.exceptions import RetrievalError
from src.observability import RERANK_LATENCY, RETRIEVAL_DOCUMENTS, get_logger
from src.rag.reranker import Reranker, create_reranker
from src.rag.types import RetrievedContext, RetrievedDocument

if typing.TYPE_CHECKING:
    from src.rag.vector_store import VectorStore
    from src.settings import RetrievalSettings

logger = get_logger(__name__)


class RetrievedEvidence(BaseModel):
    """Internal evidence DTO passed between retrieval and answer construction."""

    document: RetrievedDocument
    vector_score: float
    rerank_score: float | None = None


class RetrievalPipeline:
    """Coordinates vector retrieval and reranking into a typed context."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: typing.Any,
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
        """Return authorized, reranked evidence for a query."""

        if not query.strip():
            raise ValueError("Retrieval query must not be empty.")

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
            raise RetrievalError(f"Vector retrieval failed: {exc}") from exc

        evidence = [
            RetrievedEvidence(
                document=self._to_document(candidate.record, candidate.score),
                vector_score=candidate.score,
            )
            for candidate in candidates
            if self._is_authorized(candidate.record, owner_id)
        ]

        if RETRIEVAL_DOCUMENTS:
            RETRIEVAL_DOCUMENTS.labels(tenant_id=tenant_id).observe(len(evidence))

        if not evidence:
            logger.info("retrieval_empty", extra={"query_length": len(query), "tenant_id": tenant_id})
            return RetrievedContext(query=query, documents=[], search_type="vector")

        if self._settings.rerank.enabled:
            evidence = self._apply_reranking(query, evidence)

        documents = self._to_context_documents(evidence[:limit])
        context = RetrievedContext(query=query, documents=documents, search_type="vector")
        logger.info(
            "retrieval_completed",
            tenant_id=tenant_id,
            candidates=fetch_k,
            retrieved=len(documents),
            reranked=self._settings.rerank.enabled,
        )
        return context

    def _apply_reranking(self, query: str, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
        with RERANK_LATENCY.time() if RERANK_LATENCY else _nullcontext():
            results = self._reranker.rerank(
                query, [entry.document for entry in evidence], top_k=len(evidence)
            )

        reranked: list[RetrievedEvidence] = []
        for result in results:
            for entry in evidence:
                if entry.document.chunk_id == result.document.chunk_id:
                    reranked.append(
                        RetrievedEvidence(
                            document=entry.document,
                            vector_score=entry.vector_score,
                            rerank_score=result.score,
                        )
                    )
                    break
        reranked.sort(key=lambda item: item.rerank_score or item.vector_score, reverse=True)
        return reranked

    def _to_context_documents(
        self, evidence: typing.Sequence[RetrievedEvidence]
    ) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []
        for entry in evidence:
            base = entry.document.model_dump()
            base.pop("rerank_score", None)
            documents.append(
                RetrievedDocument(
                    rerank_score=entry.rerank_score,
                    **base,
                )
            )
        return documents

    @staticmethod
    def _is_authorized(record: typing.Any, owner_id: str | None) -> bool:
        if owner_id and getattr(record, "owner_id", None):
            return record.owner_id == owner_id
        return True

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


class _nullcontext:
    def __enter__(self) -> _nullcontext:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


__all__ = ["RetrievalPipeline", "RetrievedEvidence"]

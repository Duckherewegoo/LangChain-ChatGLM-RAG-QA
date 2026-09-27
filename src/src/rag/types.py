"""Retrieval domain types."""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from langchain_core.documents import Document as LangChainDocument


class RetrievedDocument(BaseModel):
    """Retrieved evidence with tenant-aware access metadata."""

    document_id: str
    chunk_id: str
    title: str = ""
    source: str
    content: str
    score: float = 0.0
    rerank_score: float | None = None
    page: int | None = None
    tenant_id: str = "default"
    owner_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, typing.Any] = Field(default_factory=dict)

    def citation(self) -> str:
        """Return a compact citation string for answer post-processing."""

        title = self.title.strip() or self.source
        location = f":{self.page}" if isinstance(self.page, int) and self.page > 0 else ""
        return f"{title}{location}"

    def to_langchain_document(self) -> LangChainDocument:
        """Create a LangChain document while retaining original metadata."""

        from langchain_core.documents import Document as LangChainDocument

        return LangChainDocument(
            page_content=self.content,
            metadata={
                "document_id": self.document_id,
                "chunk_id": self.chunk_id,
                "source": self.source,
                "title": self.title,
                "score": self.score,
                "rerank_score": self.rerank_score,
                "page": self.page,
                "tenant_id": self.tenant_id,
                "owner_id": self.owner_id,
                "tags": list(self.tags),
                **self.metadata,
            },
        )


class RetrievedContext(BaseModel):
    """Authorized retrieval result ready for prompt construction."""

    query: str
    documents: list[RetrievedDocument] = Field(default_factory=list)
    search_type: str = "vector"
    truncated: bool = False

    def is_empty(self) -> bool:
        """Return True when no documents survived retrieval and reranking."""

        return not self.documents

    def distinct_sources(self) -> list[str]:
        """Return source paths in their original retrieval order."""

        seen: list[str] = []
        for document in self.documents:
            if document.source not in seen:
                seen.append(document.source)
        return seen

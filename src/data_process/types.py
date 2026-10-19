"""文档摄入管线的数据类型定义。"""

from __future__ import annotations

import typing
from pydantic import BaseModel, Field

from src.rag.vector_store import VectorRecord


class Chunk(BaseModel):
    """摄入管线产出的标准化文本块。"""

    model_config = {"frozen": True}

    document_id: str
    chunk_id: str
    text: str
    title: str = ""
    source: str
    tenant_id: str = "default"
    page: int | None = None
    chunk_index: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, typing.Any] = Field(default_factory=dict)

    def to_vector_record(self, owner_id: str | None = None) -> VectorRecord:
        """转换为向量存储记录，保留所有元数据。

        注意：chunk_index 由系统生成，业务 metadata 中若存在同名键不得覆盖它。
        """
        merged_metadata = {**self.metadata, "chunk_index": self.chunk_index}
        return VectorRecord(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            text=self.text,
            title=self.title,
            source=self.source,
            tenant_id=self.tenant_id,
            owner_id=owner_id,
            page=self.page,
            tags=list(self.tags),
            metadata=merged_metadata,
        )


class IngestStatistics(BaseModel):
    """一次摄入执行的结果。"""

    document_id: str
    source: str
    scanned: int
    chunks_created: int
    chunks_stored: int
    bytes_size: int
    skipped: bool = False
    reason: str | None = None
    errors: list[dict[str, str]] = Field(default_factory=list)


__all__ = ["Chunk", "IngestStatistics"]

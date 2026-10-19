"""检索领域类型定义。"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from langchain_core.documents import Document as LangChainDocument


class RetrievedDocument(BaseModel):
    """带租户感知访问元数据的检索证据。

    设计为不可变模型：属性重新赋值会被 Pydantic 拒绝，防止在流水线
    （重排、生成）中被意外替换。注意：tags 和 metadata 的内容仍可变，
    调用方应将它们视为只读，如需修改请使用 model_copy(update=...)。
    """

    model_config = {"frozen": True}

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
        """返回用于答案后处理的紧凑引用字符串。"""
        title = self.title.strip() or self.source
        location = f":{self.page}" if isinstance(self.page, int) and self.page > 0 else ""
        return f"{title}{location}"

    def to_langchain_document(self) -> LangChainDocument:
        """创建一个保留原始元数据的 LangChain 文档对象。

        优先使用核心字段，业务自定义元数据仅作为补充，防止自定义字段覆盖核心身份。
        """
        from langchain_core.documents import Document as LangChainDocument

        # 先构造核心元数据，再用业务 metadata 补充尚未存在的键
        merged_metadata = {
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
        }
        for key, value in self.metadata.items():
            # 核心字段不允许被自定义 metadata 覆盖
            merged_metadata.setdefault(key, value)

        return LangChainDocument(
            page_content=self.content,
            metadata=merged_metadata,
        )


class RetrievedContext(BaseModel):
    """用于提示词构建的授权检索结果快照。"""

    model_config = {"frozen": True}

    query: str
    documents: list[RetrievedDocument] = Field(default_factory=list)
    search_type: str = "vector"
    truncated: bool = False
    """当上下文超出预算被截断时为 True。当前流水线未实现截断，字段预留给后续扩展。"""

    def is_empty(self) -> bool:
        """当检索和重排后没有保留任何文档时返回 True。"""
        return not self.documents

    def distinct_sources(self) -> list[str]:
        """按原始检索顺序返回去重后的来源路径列表。"""
        seen: list[str] = []
        for document in self.documents:
            if document.source not in seen:
                seen.append(document.source)
        return seen

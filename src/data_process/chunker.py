"""文档切分器。"""

from __future__ import annotations

import hashlib
import os
import re
import typing

from rag.vector_store import VectorRecord
from .types import Chunk


class DocumentChunker:
    """按固定窗口与段落边界切分文本，生成稳定标识的 chunk。"""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        min_characters: int = 50,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正数。")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须在 [0, chunk_size) 范围内。")
        if min_characters <= 0:
            raise ValueError("min_characters 必须为正数。")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_characters = min_characters

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    def split(
        self,
        text: str,
        *,
        source: str,
        title: str = "",
        document_id: str | None = None,
        tenant_id: str = "default",
        tags: typing.Iterable[str] | None = None,
        metadata: dict[str, typing.Any] | None = None,
    ) -> list[Chunk]:
        """将文本切分为标准化 chunk 列表。"""
        normalized = normalize_whitespace(text)
        if not normalized.strip():
            return []

        # 关键：document_id 必须由系统统一计算，不能依赖调用方传入的歧义值
        doc_id = document_id or compute_document_id(source, normalized)

        sections = self._split_sections(normalized)
        chunks: list[Chunk] = []
        for section in sections:
            if len(section) < self._min_characters:
                continue
            chunks.extend(
                self._chunk_section(
                    section,
                    doc_id,
                    index_start=len(chunks),
                    title=title,
                    source=source,
                    tenant_id=tenant_id,
                    tags=tags,
                    metadata=metadata,
                )
            )
        return chunks

    def split_langchain_documents(
        self,
        documents: typing.Iterable[typing.Any],
    ) -> list[Chunk]:
        """将 LangChain 文档列表转换为项目 Chunk。"""
        results: list[Chunk] = []
        for document in documents:
            metadata = dict(document.metadata or {})
            raw_tags = metadata.pop("tags", None)
            if isinstance(raw_tags, str):
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            elif isinstance(raw_tags, (list, tuple)):
                tags = [str(t) for t in raw_tags]
            else:
                tags = []

            results.extend(
                self.split(
                    document.page_content,
                    source=str(metadata.get("source", "")),
                    title=str(metadata.get("title", "")),
                    tenant_id=str(metadata.get("tenant_id", "default")),
                    tags=tags,
                    metadata=metadata,
                )
            )
        return results

    def _chunk_section(
        self,
        section: str,
        document_id: str,
        *,
        index_start: int,
        title: str,
        source: str,
        tenant_id: str,
        tags: typing.Iterable[str] | None,
        metadata: dict[str, typing.Any] | None,
    ) -> list[Chunk]:
        if len(section) <= self._chunk_size:
            return [
                self._create_chunk(
                    document_id, index_start, section,
                    title=title, source=source, tenant_id=tenant_id,
                    tags=tags, metadata=metadata,
                )
            ]

        chunks: list[Chunk] = []
        stride = max(1, self._chunk_size - self._chunk_overlap)
        position = 0
        index = index_start
        while position < len(section):
            window = section[position:position + self._chunk_size]
            chunks.append(
                self._create_chunk(
                    document_id, index, window,
                    title=title, source=source, tenant_id=tenant_id,
                    tags=tags, metadata=metadata,
                )
            )
            index += 1
            position += stride
            if position >= len(section):
                break
            if len(section) - position <= self._chunk_size:
                chunks.append(
                    self._create_chunk(
                        document_id, index, section[position:],
                        title=title, source=source, tenant_id=tenant_id,
                        tags=tags, metadata=metadata,
                    )
                )
                break
        return chunks

    def _create_chunk(
        self,
        document_id: str,
        index: int,
        text: str,
        *,
        title: str,
        source: str,
        tenant_id: str,
        tags: typing.Iterable[str] | None,
        metadata: dict[str, typing.Any] | None,
    ) -> Chunk:
        return Chunk(
            document_id=document_id,
            chunk_id=f"{document_id}#{index}",
            text=text,
            title=title,
            source=source,
            tenant_id=tenant_id,
            chunk_index=index,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )

    def _split_sections(self, text: str) -> list[str]:
        """按段落边界切分，再合并到接近 chunk_size。"""
        parts = re.split(r"\n{2,}", text)
        sections: list[str] = []
        buffer = ""
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue
            candidate = f"{buffer}\n{cleaned}".strip() if buffer else cleaned
            if len(candidate) <= self._chunk_size:
                buffer = candidate
            else:
                if buffer:
                    sections.append(buffer)
                buffer = cleaned
        if buffer:
            sections.append(buffer)
        return sections


def normalize_whitespace(text: str) -> str:
    """统一换行符并去掉行尾空白。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines())


def compute_document_id(source: str, content: str | None = None) -> str:
    """基于路径与内容生成稳定的文档标识。

    无论是否有 content，同一文件在同一路径下必须生成同一个 document_id，
    否则 delete_by_document 会失效。
    """
    digest = hashlib.sha256()
    digest.update(os.path.abspath(source).encode("utf-8"))
    if content is not None:
        digest.update(b"\x00")
        digest.update(content.encode("utf-8"))
    return digest.hexdigest()[:24]


__all__ = ["DocumentChunker", "compute_document_id", "normalize_whitespace"]

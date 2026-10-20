"""文档摄入管线。"""

from __future__ import annotations

import logging
import os
import typing

from exceptions import DocumentLoadError, IngestError
from .chunker import DocumentChunker, compute_document_id, normalize_whitespace
from .loaders import DocumentLoader
from .types import IngestStatistics

logger = logging.getLogger(__name__)


class IngestPipeline:
    """端到端摄入协调器。"""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: DocumentChunker,
        vector_store: typing.Any,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._vector_store = vector_store

    def ingest_path(
        self,
        path: str | os.PathLike[str],
        *,
        tenant_id: str = "default",
        owner_id: str | None = None,
        embedder: typing.Any,
        title: str | None = None,
        tags: typing.Iterable[str] | None = None,
    ) -> IngestStatistics:
        """遍历路径下的文件，逐个执行解析、切分、嵌入、写入。

        单文件失败不阻断其他文件；失败信息被记录在 errors 中返回。
        """
        files = (
            self._loader.scan(path) if os.path.isdir(path) else [os.path.abspath(path)]
        )
        if not files:
            raise IngestError(f"路径下未发现支持的文档: {path}")

        total_chunks = 0
        total_stored = 0
        total_bytes = 0
        errors: list[dict[str, str]] = []
        # 顶层 document_id 仅在有单个文件时有意义
        root_document_id = (
            compute_document_id(files[0]) if len(files) == 1 else "batch"
        )

        for file_path in files:
            try:
                text, metadata = self._loader.load(file_path)
            except DocumentLoadError as exc:
                logger.warning(f"文档加载失败，已跳过 | path={file_path} | error={exc}")
                errors.append({"path": str(file_path), "stage": "load", "error": str(exc)})
                continue

            total_bytes += int(metadata.get("bytes_size", 0))
            doc_id = compute_document_id(file_path, normalize_whitespace(text))

            try:
                chunks = self._chunker.split(
                    text,
                    source=file_path,
                    title=title or os.path.basename(file_path),
                    tenant_id=tenant_id,
                    document_id=doc_id,
                    tags=tags,
                )
            except Exception as exc:
                logger.warning(f"文档切分失败 | path={file_path} | error={exc}")
                errors.append({"path": str(file_path), "stage": "chunk", "error": str(exc)})
                continue

            if not chunks:
                continue

            records = [chunk.to_vector_record(owner_id=owner_id) for chunk in chunks]
            try:
                embeddings = embedder.embed_documents([r.text for r in records])
                stored = self._vector_store.add(records, embeddings)
            except Exception as exc:
                logger.warning(f"文档写入失败 | path={file_path} | error={exc}")
                errors.append({"path": str(file_path), "stage": "store", "error": str(exc)})
                continue

            total_chunks += len(chunks)
            total_stored += stored
            logger.info(f"文档摄入成功 | path={file_path} | chunks={len(chunks)}")

        return IngestStatistics(
            document_id=root_document_id,
            source=str(path),
            scanned=len(files),
            chunks_created=total_chunks,
            chunks_stored=total_stored,
            bytes_size=total_bytes,
            skipped=False,
            errors=errors,
        )

    def remove_document(self, document_id: str, tenant_id: str = "default") -> int:
        """删除某个文档的所有 chunk。"""
        return self._vector_store.delete_by_document(document_id, tenant_id)


__all__ = ["IngestPipeline"]

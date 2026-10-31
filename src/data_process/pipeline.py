"""文档摄入管线。"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable

from exceptions import DocumentLoadError, IngestError
from observability import get_logger
from .chunker import DocumentChunker, compute_document_id, normalize_whitespace
from .loaders import DocumentLoader
from .types import IngestStatistics

if TYPE_CHECKING:  # 仅用于类型检查，避免运行时循环 import
    from model import EmbeddingModel
    from rag.vector_store import VectorStore

logger = get_logger(__name__)


ErrorEntry = dict[str, str]


class IngestPipeline:
    """端到端摄入协调器。"""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: DocumentChunker,
        vector_store: "VectorStore",
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
        # embedder 作为方法参数而非构造参数：容器复用同一实例，测试可替换。
        embedder: "EmbeddingModel",
        title: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> IngestStatistics:
        """遍历路径下的文件，逐个执行解析、切分、嵌入、写入。

        单文件失败不阻断其他文件；失败信息被记录在 errors 中返回。

        安全说明：path 未做白名单校验，调用方需保证传入可信路径。

        幂等性说明：同一 path 重复摄入时，是否覆盖取决于 vector_store.add
        的语义（upsert 或新增）；本管道自身不做"先删后写"。

        已知问题（P0-5）：构造时未注入 keyword_index，混合检索的 keyword
        分支永远返回空。修复需 container 层把 _keyword_index 注入本类，
        并在 chunk 成功后写入——属跨文件改动。

        已知问题（B6）：errors 在本函数内累积，但 API 层
        （api/schemas.IngestResponseBody）未暴露该字段，用户看不到失败详情。
        """
        # tags 可能是一次性迭代器（生成器）；入口规范化为 tuple，避免下游
        # 多次迭代时第二次拿到空。空列表 / None 统一为 None。
        tags = tuple(tags) if tags else None

        files = (
            self._loader.scan(path) if os.path.isdir(path) else [os.path.abspath(path)]
        )
        if not files:
            raise IngestError(f"路径下未发现支持的文档: {path}")

        single_file = len(files) == 1
        # 单文件：用循环内算出的实际 doc_id；批量：无单个可删实体，用 "batch"
        # 占位（此前单文件返回 compute_document_id(files[0])（无文本），与循环内
        # 写入的 doc_id（含文本）不一致，导致 API 返回的 ID 删不掉实际数据）。
        result_document_id = "batch"

        total_created = 0
        total_stored = 0
        total_bytes = 0
        errors: list[ErrorEntry] = []

        for file_path in files:
            try:
                text, metadata = self._loader.load(file_path)
            except DocumentLoadError as exc:
                self._record_error(errors, file_path, "load", exc, "文档加载失败，已跳过")
                continue

            total_bytes += int(metadata.get("bytes_size", 0))
            doc_id = compute_document_id(file_path, normalize_whitespace(text))
            if single_file:
                result_document_id = doc_id

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
                self._record_error(errors, file_path, "chunk", exc, "文档切分失败")
                continue

            if not chunks:
                # 空文档（如全为图片 / 二进制）不应静默跳过：调用方需要区分
                # "全空"与"正常"，否则 errors=[] + chunks_stored=0 无法解读。
                errors.append(
                    {"path": str(file_path), "stage": "chunk", "error": "no chunks produced"}
                )
                continue

            # 切分成功即计入 chunks_created；与"成功存储"分开统计，
            # 后者才对应 chunks_stored。
            total_created += len(chunks)

            # to_vector_record 可能因字段缺失 / 序列化失败抛错——必须包进 try，
            # 否则会崩掉整批并丢失已累积的统计。
            try:
                records = [chunk.to_vector_record(owner_id=owner_id) for chunk in chunks]
            except Exception as exc:
                self._record_error(errors, file_path, "record", exc, "向量记录构造失败")
                continue

            # embedding 与写入分开 try：前者是模型侧故障，后者是向量库侧故障，
            # 统一记为 stage="store" 会掩盖运维信号。
            try:
                embeddings = embedder.embed_documents([r.text for r in records])
            except Exception as exc:
                self._record_error(errors, file_path, "embed", exc, "向量嵌入失败")
                continue

            try:
                stored = self._vector_store.add(records, embeddings)
            except Exception as exc:
                self._record_error(errors, file_path, "store", exc, "向量写入失败")
                continue

            total_stored += stored
            logger.info(
                "文档摄入成功 | path=%s | chunks=%d", file_path, len(chunks)
            )

        return IngestStatistics(
            document_id=result_document_id,
            source=str(path),
            scanned=len(files),
            chunks_created=total_created,
            chunks_stored=total_stored,
            bytes_size=total_bytes,
            # 注（M7）：skipped 字段在 schema 中保留但未实现语义；
            # 当前恒为 False。如需区分"整批全跳过"，可从 errors + chunks 推断。
            skipped=False,
            errors=errors,
        )

    def remove_document(self, document_id: str, tenant_id: str = "default") -> int:
        """删除某个文档的所有 chunk。"""
        return self._vector_store.delete_by_document(document_id, tenant_id)

    @staticmethod
    def _record_error(
        errors: list[ErrorEntry],
        path: str | os.PathLike[str],
        stage: str,
        exc: Exception,
        message: str,
    ) -> None:
        """记录一条摄入错误，同时打 warning 日志。"""
        logger.warning("%s | path=%s | error=%s", message, path, exc)
        errors.append({"path": str(path), "stage": stage, "error": str(exc)})


__all__ = ["IngestPipeline"]
